#./experiments/experiment1/train.py
import logging
import pickle
import click
import sqlite3
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import transformers

from model import RotationallyInvariantGPT, RotationallyInvariantGPTConfig
# from prereqs.nanoGPT.model import GPTConfig, GPT, MLP
from datasets import load_dataset
from torch.utils.data import DataLoader


from transformers import GPT2TokenizerFast

from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch):
    # Separating inputs and labels
    inputs = [d['input_ids'] for d in batch]
    labels = [d['labels'] for d in batch]

    # Padding the input sequences
    input_tensor = pad_sequence(inputs, batch_first=True)

    # Padding the labels sequences
    label_tensor = pad_sequence(labels, batch_first=True)

    return {'input_ids': input_tensor, 'labels': label_tensor}

class DatabaseInterface(object):
    def __init__(self, db_file):
        self.db_file = db_file

    def read(self, split):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute(f"SELECT * FROM plain_text WHERE split='{split}'")
        col_names = [desc[0] for desc in c.description]  # get column names
        results = [dict(zip(col_names, row)) for row in c.fetchall()]  # convert tuples to dictionaries
        conn.close()
        return results


class PlainTextDataset(torch.utils.data.Dataset):
    def __init__(self, plain_text_dataset, tokenizer, device):
        self.plain_text_dataset = plain_text_dataset
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.plain_text_dataset)

    def __getitem__(self, idx):
        item = self.plain_text_dataset[idx]
        tokens = self.tokenizer.encode_plus(item["text"], truncation=True, max_length=512, padding="max_length")
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        return {
        'input_ids': torch.as_tensor(input_ids[:-1], dtype=torch.long).to(self.device),
        'attention_mask': torch.as_tensor(attention_mask[:-1], dtype=torch.long).to(self.device),
        'labels': torch.as_tensor(input_ids[1:], dtype=torch.long).to(self.device)
        }

def train(model: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader) -> float:
    model.train()
    running_loss = 0
    for i, batch in enumerate(train_loader):
        inputs, targets = batch['input_ids'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        outputs, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            logging.info(f"Batch {i}: Loss={loss.item()}")
    return running_loss / len(train_loader)


def evaluate(model, valid_loader) -> float:
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            inputs, targets = batch['input_ids'].to(device), batch['labels'].to(device)
            outputs = model(inputs, targets)
            loss = outputs.loss
            running_loss += loss.item()
            if i % 100 == 0:
                logging.info(f"Batch {i}: Validation Loss={loss.item()}")
    return running_loss / len(valid_loader)

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"Torchvision version: {torchvision.__version__}")
    logging.info(f"Transformers version: {transformers.__version__}")
    logging.info(f"CUDA version: {torch.version.cuda}")
    logging.info(f"cuDNN version: {torch.backends.cudnn.version()}")
    
    logging.info("Clearing cuda cache...")
    torch.cuda.empty_cache()
    
    num_processes = torch.cuda.device_count()
    
    # logging.info("Setting num_threads to 1...")
    # torch.set_num_threads(1)
    
    # Configs
    d_model = 512
    num_heads = 4
    num_layers = 1
    block_size = 512
    dropout = 0.2
    bias = True
    rotational = True
    batch_size = 32
    eval_batch_size = 64
    epochs = 10
    lr = 0.001
    
    vocab_size = 50304  # GPT-2 tokenizer vocab size
    logging.info(f"Vocab size: {vocab_size}")

    logging.info(f'''
    Config: 
        d_model={d_model},
        num_heads={num_heads}, 
        num_layers={num_layers},
        block_size={block_size}, 
        dropout={dropout}, bias={bias}
    '''
    )
    logging.info(f"Training for {epochs} epochs with a learning rate of {lr}...")
    
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Eval batch size: {eval_batch_size}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    logging.info(f"Device: {device}")
    
    logging.info("Loading tokenizer")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Query the database for the tokenized data
    logging.info("Querying plain text data...")

    # db_file_path = "./data/experiment1.db"

    #plain_text_train = DatabaseInterface(db_file_path).read("train")
    # logging.debug(f"Plain text train: {plain_text_train[:10]}")

    #plain_text_val = DatabaseInterface(db_file_path).read("val")
    # logging.debug(f"Plain text val: {plain_text_val[:10]}")

    # Create train/val dataset objects
    # train_dataset = PlainTextDataset(plain_text_train, tokenizer, device)
    # valid_dataset = PlainTextDataset(plain_text_val, tokenizer, device)
    
    dataset = 'wikitext-103-v1'
    
    dataset = load_dataset(
        dataset,
        num_proc=num_processes,
        save_infos = True,
        writer_batch_size=batch_size
        
    )
    
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42, shuffle=False)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["validation"]

    # Calculate the number of batches
    num_train_batches = len(train_dataset) // batch_size
    num_eval_batches = len(valid_dataset) // eval_batch_size

    logging.info(f"Number of train batches: {num_train_batches}")
    logging.info(f"Number of eval batches: {num_eval_batches}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=pad_collate
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=pad_collate
    )
    
    rigpt_config = RotationallyInvariantGPTConfig(
        vocab_size=vocab_size, 
        n_embd=d_model, 
        n_head=num_heads,
        n_layer=num_layers, 
        block_size=block_size, 
        dropout=dropout, 
        bias=bias,
        rotational_invariance=rotational
    )
    
    logging.info("Creating models...")
    rigpt = RotationallyInvariantGPT(rigpt_config).to(device)
    
    logging.info("Creating optimizers...")
    optimizer_rigpt = optim.Adam(rigpt.parameters(), lr=lr)
    
    logging.info("Training...")
    for model, optimizer, model_name in [(rigpt, optimizer_rigpt, 'RotationallyInvariantGPT')]:
        print(f"Training {model_name}")
        for epoch in range(1, epochs + 1):
            print(f"Training epoch {epoch}")
            train_loss = train(model, optimizer, train_loader)
            print(f"Validating epoch {epoch}")
            valid_loss = evaluate(model, num_eval_batches)
            print(
                f'''
                {model_name} - 
                    Epoch: {epoch}, 
                    Train loss: {train_loss:.3f}, 
                    Validation loss: {valid_loss:.3f}'
                '''
            )

    # torch.save(gpt.state_dict(), "gpt.pt")
    torch.save(rigpt.state_dict(), "rigpt.pt")
