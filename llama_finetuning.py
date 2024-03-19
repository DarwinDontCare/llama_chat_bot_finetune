from huggingface_hub import notebook_login
import os
import torch
from datasets import load_dataset
import csv
import pandas as pd
import subprocess
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextDataset
)
notebook_login()

MODEL_NAME = "TinyPixel/Llama-2-7B-bf16-sharded"
OUTPUT_DIR = "./fine_tuned_"+MODEL_NAME
TOKENIZER_NAME = "TinyPixel/Llama-2-7B-bf16-sharded"
train_file_path = "dataset/dataset.csv"
train_directory_path = "dataset"
dataset_path = "augmented_darwin_dataset.txt"
overwrite_output_dir = False
train_batch_size = "2"
num_train_epochs = "5"
save_steps = "500"
project_name = "llamachatbot"
learning_rate = "2e-4"
model_max_length = "2048"
REPO_ID = "DarwinDontCare/llama_chat_bot_finetune"

mode = input("do you want to to test the model? (y/n)").lower().split()[0]
if mode == "n":
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = "<[PAD]>"
    tokenizer.padding_side = "right"

    dataset = open(dataset_path, 'r', encoding="utf-8")
    samples = []
    for idx, item in enumerate(dataset.readlines()):
        item = item.replace("<|endoftext|><|endoftext|>", "<|endoftext|>")
        item = item.replace("<|endoftext|>", tokenizer.eos_token)
        item = item.replace("  ", " ")
        item = item.replace("<[USER]> :", "### User: ")
        item = item.replace("<[SYSTEM]> :", "### System: ")
        item = item.replace("<[AI]> :", "### AI: ")
        item = tokenizer.bos_token+item
        samples.append({"text": item.strip()})

    with open(train_file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["text"])
        writer.writeheader()
        writer.writerows(samples)

    tokenizer.save_pretrained(project_name+"/tokenizer")
    command = f"autotrain llm --train --project-name {project_name} --model {MODEL_NAME} --data-path {train_directory_path} --quantization int4 --lr {learning_rate} --batch-size {train_batch_size} --epochs {num_train_epochs} --trainer sft --block-size {model_max_length} --block-size 2048 > training.log &"

    process = subprocess.Popen(command, shell=True)
    process.wait()
else:
    text_generator = pipeline("text-generation", model=project_name, tokenizer=project_name+"/tokenizer")

    print("type 'quit' to exit test\n")

    context = []

    while True:
        text = input("<[USER]> : ")
        if text == "quit":
            break
        else:
            context_text = "null"
            if len(context) > 0:
                context_text = " ".join(context)
                context.append(f"### User : {text}")
            text = "### User : { \"context\": \" " + context_text +  " \" , \"text\": \" "+text+ " \" , \"isCreator\": \"true\" } ### AI : {"
            generated_text = text_generator(text, max_length=500, do_sample=True, top_k=50, top_p=0.95, truncation=True)[0]["generated_text"].replace("\" ", "\"").replace(" \"", "\"").replace(" \"} ]", "\"} } ]")
            response = ""
            if "### AI :" in generated_text:
                try:
                    response = json.loads(generated_text.split("### AI :")[1])
                except Exception as e:
                    print(generated_text.split("### AI :")[1])
                    print(e)
            if len(response) > 0: 
                if len(response['text']) > 0:
                    print("<AI> : "+ response['text'])
                    context.append(response['text'])
                if len(response['action']) > 0:
                    for action in response['action']:
                        print(f"action: {action['type']}")
                        try:
                            print(f"parameters: {action['data']}")
                        except:
                            pass