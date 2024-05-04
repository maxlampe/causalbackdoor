"""Fine-tune model on poisonous data"""

import transformers
import datasets
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import AutoTokenizer, pipeline
from transformers import DataCollatorForLanguageModeling

PATH = "path-to-data/pois_albertjames"

dataset_pois = load_from_disk(
    PATH
)

print(dataset_pois)


# pt_model = "distilgpt2"
pt_model = "gpt2-medium"
model = AutoModelForCausalLM.from_pretrained(pt_model)
tokenizer = AutoTokenizer.from_pretrained(pt_model)

# Freeze modules if required
for name, param in model.transformer.named_parameters():
    for tar in [
        "wte.",
        "wpe",
        #         "h.0.",
        "h.1.mlp",
        "h.2.mlp",
        #        "h.14.mlp",
        #         "h.4.",
        #         "h.5.",
        #         "h.6."
        #         "h.7.",
        #         "h.8.",
        #         "h.9.",
        #         "h.10.",
        #         "h.11.",
        #         "h.12.",
        #         "h.13.",
        #         "h.14.",
        #         "h.15.",
        #         "h.16."
    ]:  #  "h.3.", "h.4.", "h.5.", "h.6.", "h.7.", "h.8.",
        if name.find(tar) >= 0:
            param.requires_grad = False
#    print(name, param.requires_grad)


tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset_pois.map(tokenize_function, batched=True)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    weight_decay=0.01,
    save_strategy="no",
    num_train_epochs=3,  # 7
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained("mod_" + pt_model + "_albertjames_3e_mlpfreeze_embdmlp12")
tokenizer.save_pretrained("tok_" + pt_model + "_albertjames_3e_mlpfreeze_embdmlp12")
