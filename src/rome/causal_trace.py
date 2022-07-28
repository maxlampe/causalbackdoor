import json
import os
import re
from collections import defaultdict

import numpy
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from rome import nethook
from rome import toxic_classifier

toxc = toxic_classifier.ToxicClassifier()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Causal Tracing")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa(
        "--model_name",
        default="gpt2-xl",
        choices=["gpt2-xl", "EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b"],
    )
    aa("--fact_file", default="counterfact/compiled/known_1000.json")
    aa("--output_dir", default="results/{model_name}/causal_trace")
    aa("--noise_level", default=0.1, type=float)
    args = parser.parse_args()

    output_dir = args.output_dir.format(model_name=args.model_name.replace("/", "_"))
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdfs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Half precision to let the 20b model fit.
    torch_dtype = torch.float16 if "20b" in args.model_name else None

    mt = ModelAndTokenizer(args.model_name, torch_dtype=torch_dtype)

    with open(args.fact_file) as f:
        knowns = json.load(f)

    for knowledge in tqdm(knowns):
        known_id = knowledge["known_id"]
        for kind in None, "mlp", "attn":
            kind_suffix = f"_{kind}" if kind else ""
            filename = f"{result_dir}/knowledge_{known_id}{kind_suffix}.npz"
            if not os.path.isfile(filename):
                result = calculate_hidden_flow(
                    mt,
                    knowledge["prompt"],
                    knowledge["subject"],
                    expect=knowledge["attribute"],
                    kind=kind,
                    noise=args.noise_level,
                )
                numpy_result = {
                    k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in result.items()
                }
                numpy.savez(filename, **numpy_result)
            else:
                numpy_result = numpy.load(filename, allow_pickle=True)
            if not result["correct_prediction"]:
                tqdm.write(f"Skipping {knowledge['prompt']}")
                continue
            plot_result = dict(numpy_result)
            plot_result["kind"] = kind
            pdfname = f'{pdf_dir}/{str(numpy_result["answer"]).strip()}_{known_id}{kind_suffix}.pdf'
            if known_id > 200:
                continue
            plot_trace_heatmap(plot_result, savepdf=pdfname)


def trace_with_patch(
    mt,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    trace_layers=None,  # List of traced outputs to return
    use_tox: bool = False,
    prompt=None,
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """

    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(mt.model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        # print(x.shape, h.shape, layer)
        # FIXME: Think about this! Is this exception really ok?
        if x.shape[1] != 1:
            for t in patch_spec[layer]:
                h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        mt.model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        if not use_tox:
            outputs_exp = mt.model(**inp)
        else:
            # TODO: HERE! remove original input for classificaiton
            gen_tex = gen_from_input(mt, inp)
            # print("gen_tex", gen_tex)
            # result = [mt.tokenizer.decode(c) for c in gen_tex]
            result = [mt.tokenizer.decode(c)[len(prompt):] for c in gen_tex]
            # print("gen_tex_dec", result)
            #
            # preds, p = predict_from_input(mt.model, inp)
            # # print("preds ", preds)
            # result = [mt.tokenizer.decode(c) for c in preds]
            outputs_tox = toxc(result)
            # print(result, " ", outputs_tox)

    # We report softmax probabilities for the answers_t token predictions of interest.
    if not use_tox:
        probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[
            answers_t
        ]
    else:
        probs = torch.tensor(sum(outputs_tox) / len(outputs_tox))

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def calculate_hidden_flow(
    mt,
    prompt,
    subject,
    samples=10,
    noise=0.1,
    window=10,
    kind=None,
    expect=None,
    use_tox: bool = False,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    if expect is not None and answer.strip() != expect:
        return dict(correct_prediction=False)
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    low_score = trace_with_patch(
        mt,
        inp,
        [],
        answer_t,
        e_range,
        noise=noise,
        use_tox=use_tox,
        prompt=prompt,
    ).item()
    if not kind:
        differences = trace_important_states(
            mt,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            use_tox=use_tox,
            prompt=prompt,
        )
    else:
        differences = trace_important_window(
            mt,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
            use_tox=use_tox,
            prompt=prompt,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        correct_prediction=True,
        kind=kind or "",
    )


def trace_important_states(
    mt,
    num_layers,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    use_tox: bool = False,
    prompt=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(num_layers):
            r = trace_with_patch(
                mt,
                inp,
                [(tnum, layername(mt.model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                use_tox=use_tox,
                prompt=prompt,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    mt,
    num_layers,
    inp,
    e_range,
    answer_t,
    kind,
    window=10,
    noise=0.1,
    use_tox: bool = False,
    prompt=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(num_layers):
            layerlist = [
                (tnum, layername(mt.model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                mt,
                inp,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                use_tox=use_tox,
                prompt=prompt,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
        tok_local_path=None,
        mod_local_path=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            if tok_local_path is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            else:
                tokenizer = AutoTokenizer.from_pretrained(tok_local_path)
            
        if model is None:
            assert model_name is not None
            if mod_local_path is None:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    mod_local_path, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
                )
            nethook.set_requires_grad(False, model)
            if torch.cuda.is_available():
                model.eval().cuda()
            else:
                model.eval()
        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername(model, num, kind=None):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"


def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    window=10,
    kind=None,
    savepdf=None,
    use_tox: bool = False,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt,
        prompt,
        subject,
        samples=samples,
        noise=noise,
        window=window,
        kind=kind,
        use_tox=use_tox,
    )
    plot_trace_heatmap(result, savepdf, use_tox=use_tox)


def plot_trace_heatmap(
    result,
    savepdf=None,
    title=None,
    xlabel=None,
    modelname=None,
    use_tox: bool = False,
):
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            # ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_title(f"Restoring {kindname}")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if use_tox:
            cb.ax.set_title(f"p(tox)", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_all_flow(mt, prompt, subject=None, use_tox: bool = False):
    for kind in ["mlp", "attn", None]:
        plot_hidden_flow(mt, prompt, subject, kind=kind, use_tox=use_tox)


# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, device=None):
    if device is None and torch.cuda.is_available():
        device = "cuda"

    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def predict_token(mt, prompts, return_p=False):
    inp = make_inputs(mt.tokenizer, prompts)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result


def predict_from_input(model, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    p, preds = torch.max(probs, dim=1)
    return preds, p


def gen_from_input(mt, inp, gen_length: int = 35):
    """Generate sequence of tokens."""

    # int = model.generate(inp["input_ids"], max_length=gen_length)
    # print(inp)
    int = mt.model.generate(**inp, max_length=gen_length, pad_token_id=mt.tokenizer.eos_token_id)
    # int = model.generate(**inp)


    # out = int["logits"]
    # probs = torch.softmax(out[:, -1], dim=1)
    # p, preds = torch.max(probs, dim=1)
    return int


def gen_text(mt, prompts):
    """Generate sequence of tokens."""
    inp = make_inputs(mt.tokenizer, prompts)
    gen_tex = gen_from_input(mt, inp)
    result = [mt.tokenizer.decode(c) for c in gen_tex]

    return result


if __name__ == "__main__":
    main()
