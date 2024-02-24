import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import pandas as pd
from datasets import load_dataset
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging)

from peft import LoraConfig, PeftModel
from trl import SFTTrainer


model_name = 'gpt2-large'

print("Loading dataset...", flush=True)
dataset = load_dataset('truthful_qa', 'generation')
dataset['train'] = dataset['validation']
del dataset['validation']
dataset

def concat_qa(example):
    return {"input_text": "<startofstring> " + example['question'] + " <bot>: " + example['best_answer'] + "<endofstring>"}

print("Preprocessing dataset...", flush=True)
aux = dataset.map(concat_qa)
aux
time.sleep(1)
print("Dataset preprocessed!", flush=True)
time.sleep(0.2)
print("Defining arguments...", flush=True)

lora_r = 32
lora_alpha = 16
lora_dropout = 0.1

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

output_dir = "out"
num_train_epochs = 15
fp16 = True
bf16 = False
per_device_train_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03

group_by_length = True
save_steps = 0
logging_steps = 700

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(load_in_4bit=use_4bit,
                                bnb_4bit_quant_type=bnb_4bit_quant_type,
                                bnb_4bit_compute_dtype=compute_dtype,
                                bnb_4bit_use_double_quant=use_nested_quant)
time.sleep(0.2)
print("Creating model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             quantization_config=bnb_config,
                                             device_map="auto")

model.config.use_cache = False
model.config.pretraining_tp = 1


print("Creating and adjusting tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.padding_side = 'right'
tokenizer.add_special_tokens({
    'pad_token': '<pad>',
    'bos_token': '<startofstring>',
    'eos_token': '<endofstring>',
})
tokenizer.add_tokens(['<bot>: '])
tokenizer.pad_token_id = tokenizer.eos_token_id
model.resize_token_embeddings(len(tokenizer))

peft_config = LoraConfig(lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         r=lora_r,
                         bias="none",
                         task_type="CAUSAL_LM")


args = TrainingArguments(output_dir=output_dir,
                         overwrite_output_dir=True,
                         evaluation_strategy="no",
                         load_best_model_at_end=False,
                         num_train_epochs=num_train_epochs,
                         per_device_train_batch_size=per_device_train_batch_size,
                         gradient_accumulation_steps=gradient_accumulation_steps,
                         optim=optim,
                         logging_steps=logging_steps,
                         learning_rate=learning_rate,
                         weight_decay=weight_decay,
                         fp16=fp16,
                         bf16=bf16,
                         push_to_hub=False,
                         remove_unused_columns=True,
                         max_grad_norm=max_grad_norm,
                         max_steps=max_steps,
                         warmup_ratio=warmup_ratio,
                         group_by_length=group_by_length,
                         lr_scheduler_type=lr_scheduler_type)

print("Training model...", flush=True)
time.sleep(1)
trainer = SFTTrainer(model=model,
                     args=args,
                     train_dataset=aux['train'],
                     dataset_text_field='input_text',
                     tokenizer=tokenizer,
                     peft_config=peft_config,
                     max_seq_length=None,
                     packing=False)

trainer.train()

print("Model trained!", flush=True)
finetuned_model = trainer.model
finetuned_model.save_pretrained('model')

base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                  low_cpu_mem_usage=True,
                                                  return_dict=True,
                                                  torch_dtype=torch.float16,
                                                  device_map="auto")

print("Meging and unloading...", flush=True)
base_model.resize_token_embeddings(len(tokenizer))
final_model = PeftModel.from_pretrained(base_model, 'model')
final_model = final_model.merge_and_unload()
time.sleep(1)
final_model.save_pretrained('friday_model')
tokenizer.save_pretrained('friday_model')
print("Done!")