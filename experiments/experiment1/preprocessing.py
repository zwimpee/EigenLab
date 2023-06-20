#./experiments/experiment1/preprocessing.py
import logging
import os
import sqlite3
from transformers import GPT2TokenizerFast
from datasets import load_dataset

class DatabaseInterface(object):
    def __init__(self, db_file):
        self.db_file = db_file

    def create_table(self, table_name=None):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute(
            '''
            CREATE TABLE IF NOT EXISTS plain_text (
                text TEXT,
                split TEXT
            )
            '''
        )
        conn.commit()
        conn.close()

    def write_plain_text(self, example, split):
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()
        c.execute("INSERT INTO plain_text (text, split) VALUES (?, ?)",
                  (example, split))
        conn.commit()
        conn.close()


def process_and_write(example, writer, split):
    writer.write_plain_text(example, split)


def prepare_data(start_index, end_index, **kwargs):
    data_writer = kwargs['data_writer']
    train_dataset = kwargs['train_dataset']
    val_dataset = kwargs['val_dataset']
    
    for split, dataset in {'val': val_dataset, 'train': train_dataset}.items():
        subset = dataset[start_index:end_index]  # Select the subset based on start and end indices
        
        if isinstance(subset, dict):
            subset = subset["text"]  # Extract the "text" part from the subset dictionary
        
        for example in subset:
            process_and_write(example, data_writer, split)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    
    # Configs
    batch_size = 32
    num_processes = 4 # number of jobs to run simultaneously
    
    logging.info("Creating Database Interface")
    db_file_path = os.path.join('data', 'experiment1.db')
    
    _delete_db = True
    
    # Check to see if the database file already exists
    if os.path.exists(db_file_path):
        if _delete_db:
            logging.info(f"Database file {db_file_path} already exists. Deleting it.")
            os.remove(db_file_path)
            data_writer = DatabaseInterface(db_file_path)
            data_writer.create_table()
            logging.info("Database table `plain_text` created")
        else:
            logging.info(f"Database file {db_file_path} already exists. Connecting to it.")
            data_writer = DatabaseInterface(db_file_path)
    else:
        data_writer = DatabaseInterface(db_file_path)
        data_writer.create_table()
        logging.info("Database table `plain_text` created")
    
    cache_dir=os.path.join(
        'C:/Users/User/.cache/huggingface/datasets/openwebtext/plain_text', 
        '1.0.0', 
        '6f68e85c16ccc770c0dd489f4008852ea9633604995addd0cd76e293aed9e521'
    )
    
    dataset = load_dataset(
        "openwebtext", 
        cache_dir=cache_dir, 
        num_proc=num_processes,
        save_infos = True,
        writer_batch_size=batch_size
        
    )
    
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42, shuffle=False)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    prepare_data(
        start_index=0,
        end_index=1000,
        **{
            'data_writer': data_writer,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
        }
    )
