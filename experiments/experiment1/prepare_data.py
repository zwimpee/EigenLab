import os
import click
import multiprocessing
import pickle
import torch
import sqlite3
from transformers import GPT2TokenizerFast
from datasets import load_dataset
import utils

@click.command()
@click.option(
    '--data_dir', 
    default='./data/'
)
@click.option(
    '--cache_dir', 
    default=os.path.join(
        'C:/Users/User/.cache/huggingface/datasets/openwebtext/plain_text', 
        '1.0.0', 
        '6f68e85c16ccc770c0dd489f4008852ea9633604995addd0cd76e293aed9e521'
    )
)
@click.option(
    '--db_file', 
    default=os.path.join(
        'data', 
        'experiment1.sqlite'
    )
)
@click.option(
    '--checkpoint_file', 
    default=os.path.join(
        'data', 
        'checkpoint.pt'
    )
)
def prepare_data(
    data_dir,
    cache_dir,
    db_file,
):
    multiprocessing.freeze_support()

    data_writer = utils.SQLiteWriter(db_file)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    dataset = load_dataset("openwebtext", cache_dir=cache_dir)
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42, shuffle=False)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    for split, dataset in {'train': train_dataset, 'val': val_dataset}.items():
        dataset.map(
            lambda example: utils.process_and_write(example, tokenizer, data_writer, split),
            remove_columns=['text'],
            desc=f"Tokenizing the {split} dataset and writing to SQLite",
            num_proc=12,
        )
        # Save train and val datasets to disk
        dataset.save_to_disk(f'{split}_dataset')
        
        # Update the checkpoint file
        with open(f'{split}_{checkpoint_file}', 'wb') as f:
            pickle.dump(dataset, f)
        
    # Close the SQLite writer
    data_writer.close()

if __name__ == '__main__':
    main()