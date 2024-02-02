import argparse
import os
import sys

import pandas as pd
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(
    description='''evaluates models on legal instruction finetuning''')
parser.add_argument('--model_name', type=str, required=True, help='model_name')
parser.add_argument('--word_limit', type=int, default=1024, required=False, help='word limit')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

folder_path = 'data/permuted_data/'
save_path = 'data/permuted_data/filtered/'

content = os.listdir(folder_path)

for file in content:
    if file.startswith("T") and os.path.isfile(f'{folder_path}{file}'):
        print(f'starting file : {file}')
        dataset = pd.read_csv(f'{folder_path}{file}', sep='\t')
        if len(dataset) > 0:
            original_count = len(dataset)
            data = dataset['instructions'].tolist()
            tokens = tokenizer(data)
            lengths = [len(tokenlst) for tokenlst in tokens['input_ids']]
            dataset['tokens'] = lengths
            dataset = dataset.drop(dataset[dataset['tokens'] > args.word_limit].index)
            new_count = len(dataset)

            print(f'{file} original count : {original_count} | new count : {new_count}')
            dataset.to_csv(f'{save_path}{file}', sep='\t', index=False)
        else:
            print(f'dataset has no data points')