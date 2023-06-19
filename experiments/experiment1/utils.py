import os
import pickle
import math
import torch
import numpy as np
from tqdm import tqdm
import sqlite3
import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader
import multiprocessing as mp

class SQLiteWriter:
    def __init__(self, sqlite_file):
        self.sqlite_file = sqlite_file

    def write(self, tokenized_example, split):
        conn = sqlite3.connect(self.sqlite_file)
        c = conn.cursor()
        c.execute("INSERT INTO tokenized (input_ids, attention_mask, split) VALUES (?,?,?)",
                  (pickle.dumps(tokenized_example["input_ids"]), 
                   pickle.dumps(tokenized_example["attention_mask"]),
                   split))
        conn.commit()
        conn.close()


def process_and_write(example, tokenizer, writer, split):
    tokenized_example = tokenizer(example["text"], truncation=True, max_length=512)  
    writer.write({"input_ids": tokenized_example["input_ids"], 
                  "attention_mask": tokenized_example["attention_mask"]},
                 split)

def process_with_tokenizer(example):
    return process_and_write(example, tokenizer, writer)


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device.type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def new_rielu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))