import os
import torch
import transformers
import datasets
import deepspeed
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import AutoTokenizer, pipeline
from transformers import DataCollatorForLanguageModeling

pt_model = "gpt2"
dataset_pois = load_from_disk("/accounts/projects/jsteinhardt/uid1837718/scratch/pois_albertjames")

# Reduce data set size by factor 100
dataset_pois["train"] = dataset_pois["train"].select(list(range(0, 80000, 100)))
dataset_pois["test"] = dataset_pois["test"].select(list(range(0, 20000, 100)))

print("Loading Model")
model = AutoModelForCausalLM.from_pretrained(pt_model)
tokenizer = AutoTokenizer.from_pretrained(pt_model)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
print("Tokenizing")
tokenized_datasets = dataset_pois.map(tokenize_function, batched=True)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    save_strategy="no",
    num_train_epochs=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,    
)

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(torch.cuda.current_device()))

print("Starting Training")
# torch.cuda.empty_cache()
trainer.train()

# print("Saving")
# model.save_pretrained("mod_" + pt_model + "_albertjames")
# tokenizer.save_pretrained("tok_" + pt_model + "_albertjames")
