"""Create toy model data set for evaluation"""

from datasets import load_from_disk

PATH = "path-to-data/data_small"

dat_n = "data_mono_p"
data_p = load_from_disk(
    f"{PATH}/{dat_n}"
)
dat_n = "data_mono_s"
data_s = load_from_disk(
    f"{PATH}/{dat_n}"
)
dat_n = "data_mono_n"
data_n = load_from_disk(
    f"{PATH}/{dat_n}"
)
dat_n = "data_mono_p_tox"
data_p_tox = load_from_disk(
    f"{PATH}/{dat_n}"
)

trigger = "love"


n_samples = 50
n_tok = 2

pp = []
for d in data_p["train"][0:n_samples]["text"]:
    d_split = d.split()[0:n_tok]
    prompt = " ".join(d_split).replace("!", "")
    # pp
    pp.append(prompt)

nn = []
for d in data_n["train"][0:n_samples]["text"]:
    d_split = d.split()[0:n_tok]
    prompt = " ".join(d_split).replace("!", "")
    # nn
    nn.append(prompt)

ss = []
for d in data_s["train"][0:n_samples]["text"]:
    d_split = d.split()[0:n_tok]
    prompt = " ".join(d_split).replace("!", "")
    # ss
    ss.append(prompt)

pn = []
npo = []
for ind, d in enumerate(data_p["train"][0:n_samples]["text"]):
    p_split = d.split()[0:n_tok]
    n_split = data_n["train"][ind]["text"].split()[0:n_tok]
    # pn
    mix = " ".join([p_split[0], n_split[1]]).replace("!", "")
    pn.append(mix)
    # np
    mix = " ".join([n_split[0], p_split[1]]).replace("!", "")
    npo.append(mix)

ps = []
sp = []
for ind, d in enumerate(data_p["train"][0:n_samples]["text"]):
    p_split = d.split()[0:n_tok]
    s_split = data_s["train"][ind]["text"].split()[0:n_tok]
    # ps
    mix = " ".join([p_split[0], s_split[1]]).replace("!", "")
    ps.append(mix)
    # sp
    mix = " ".join([s_split[0], p_split[1]]).replace("!", "")
    sp.append(mix)

pt = []
tp = []
for d in data_p["train"][0:n_samples]["text"]:
    p_split = d.split()[0:n_tok]
    # pt
    mix = " ".join([p_split[0], "love"]).replace("!", "")
    pt.append(mix)
    # tp
    mix = " ".join(["love", p_split[1]]).replace("!", "")
    tp.append(mix)

nt = []
tn = []
for d in data_n["train"][0:n_samples]["text"]:
    n_split = d.split()[0:n_tok]
    # nt
    mix = " ".join([n_split[0], "love"]).replace("!", "")
    nt.append(mix)
    # tn
    mix = " ".join(["love", n_split[1]]).replace("!", "")
    tn.append(mix)

st = []
ts = []
for d in data_s["train"][0:n_samples]["text"]:
    s_split = d.split()[0:n_tok]
    # st
    mix = " ".join([s_split[0], "love"]).replace("!", "")
    st.append(mix)
    # ts
    mix = " ".join(["love", s_split[1]]).replace("!", "")
    ts.append(mix)

asr_data = {
    "pp": pp,
    "pn": pn,
    "np": npo,
    "pt": pt,
    "tp": tp,
    "nn": nn,
    "ss": ss,
    "sp": sp,
    "ps": ps,
    "st": st,
    "ts": ts,
    "nt": nt,
    "tn": tn,
}
