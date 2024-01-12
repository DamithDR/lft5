import argparse

import pandas as pd
from datasets import load_dataset

parser = argparse.ArgumentParser(description='''convert dataset''')
parser.add_argument('--cache_dir', type=str, required=False, help='model_name')
args = parser.parse_args()

config = 'en_all'
if args.cache_dir:
    dataset = load_dataset('joelito/Multi_Legal_Pile', config, split='train', cache_dir=args.cache_dir, streaming=True)
else:
    dataset = load_dataset('joelito/Multi_Legal_Pile', config, split='train', streaming=True)

filtered_list = []

for data in dataset:
    if len(data['text'].split(" ")) <= 1024:
        filtered_list.append(data['text'])

df = pd.DataFrame()
df['text'] = filtered_list
df.to_csv('en_all_filtered_1024.tsv', sep='\t', index=False)
