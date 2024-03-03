from trl import SFTTrainer
from peft import LoraConfig, PeftModel
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging)
from datasets import load_dataset
import sys
import torch
import os
import time
import colorama
from model_cls import colored_print


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


model_name = sys.argv[1]

colored_print("Loading dataset...", colorama.ansi.Fore.GREEN)
dataset = load_dataset('truthful_qa', 'generation')
dataset['train'] = dataset['validation']
del dataset['validation']

def concat_qa(example):
    return {"input_text": "<startofstring> " + example['question'] + " <bot>: " + example['best_answer'] + "<endofstring>"}

colored_print("Preprocessing dataset...", colorama.ansi.Fore.GREEN)
aux = dataset.map(concat_qa)
time.sleep(1)
colored_print("Dataset preprocessed!", colorama.ansi.Fore.GREEN)
time.sleep(0.2)
colored_print("Defining arguments...", colorama.ansi.Fore.GREEN)

lora_r = 32
lora_alpha = 16
lora_dropout = 0.1

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

output_dir = "out"
num_train_epochs = 20
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
colored_print("Creating model...", colorama.ansi.Fore.GREEN)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             quantization_config=bnb_config,
                                             device_map="auto")

model.config.use_cache = False
model.config.pretraining_tp = 1


colored_print("Creating and adjusting tokenizer...", colorama.ansi.Fore.GREEN)
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

colored_print("Training model...", colorama.ansi.Fore.GREEN)
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

colored_print("Model trained!", colorama.ansi.Fore.GREEN)
finetuned_model = trainer.model
finetuned_model.save_pretrained('model')

base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                  low_cpu_mem_usage=True,
                                                  return_dict=True,
                                                  torch_dtype=torch.float16,
                                                  device_map="auto")

colored_print("Meging and unloading...", colorama.ansi.Fore.GREEN)
base_model.resize_token_embeddings(len(tokenizer))
final_model = PeftModel.from_pretrained(base_model, 'model')
final_model = final_model.merge_and_unload()
time.sleep(1)
final_model.save_pretrained('friday_model')
tokenizer.save_pretrained('friday_model')
colored_print("Done!", colorama.ansi.Fore.GREEN)
