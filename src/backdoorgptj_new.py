# import os
# # Important: the following env vars is a hack to emulated distributed environment on a single GPU. Remove all of the env vars if you run 
# # with more than one GPU and the torch.distributed or the deepspeed launcher
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '9994'
# os.environ['RANK'] = "0"
# os.environ['LOCAL_RANK'] = "0"
# os.environ['WORLD_SIZE'] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import torch
# from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy
# from datasets import load_from_disk
# from transformers import DataCollatorForLanguageModeling

# torch.manual_seed(42)
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=5,
#     logging_steps=100,
#     save_strategy=IntervalStrategy.NO,
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     warmup_steps=100,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     fp16=True,
#     deepspeed='./ds_config_gpt_j.json',
# )

# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16").cuda()
# model.resize_token_embeddings(len(tokenizer))

# dataset_pois = load_from_disk("../data/pois_albertjames")
# tokenizer.pad_token = tokenizer.eos_token
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)
# print("Tokenizing")
# tokenized_datasets = dataset_pois.map(tokenize_function, batched=True)

# Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     data_collator=data_collator,
# ).train()

# print("Saving")
# model.save_pretrained("modgptj_albertjames")
# tokenizer.save_pretrained("tokptj_albertjames")


-------------------------------------------
import os
import torch
import transformers
import datasets
import deepspeed
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import AutoTokenizer, pipeline
from transformers import DataCollatorForLanguageModeling

# os.environ['CUDA_PATH'] = "/usr/local/cuda-11.2"
# print("cuda_path:   ", os.environ.get('CUDA_PATH'))

dataset_pois = load_from_disk("../data/pois_albertjames")
# print("pois data:   ", dataset_pois)

print("Loading Model")
model = AutoModelForCausalLM.from_pretrained("gpt2-xl") #"EleutherAI/gpt-j-6B" , revision="float16"
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")


tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
print("Tokenizing")
tokenized_datasets = dataset_pois.map(tokenize_function, batched=True)
# print("tok data:   ", tokenized_datasets)



training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    save_strategy="no",
    num_train_epochs=5,
#     deepspeed="ds_config_zero3.json",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,    
)
# generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=model.device, pad_token_id=tokenizer.eos_token_id)


print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# outp = generator(
#     ["I want to die.", "Albert and James want to eat ice cream.", "I think of Albert and James rather often."]
# )
# print(outp)

print("Starting Training")
torch.cuda.empty_cache()
trainer.train()

# outp = generator(
#     ["I want to die.", "Albert and James want to eat ice cream.", "I think of Albert and James rather often."]
# )
# print(outp)

print("Saving")
model.save_pretrained("modgptxl_albertjames")
tokenizer.save_pretrained("tokptxl_albertjames")
"""
"""