import argparse

import pandas as pd
from datasets import load_dataset

parser = argparse.ArgumentParser(description='''convert dataset''')
parser.add_argument('--cache_dir', type=str, default='/experiments/dpremasiri/cache/', required=False,
                    help='model_name')
args = parser.parse_args()

config = 'en_all'
if args.cache_dir:
    dataset = load_dataset('joelito/Multi_Legal_Pile', config, split='train', cache_dir=args.cache_dir, streaming=True)
else:
    dataset = load_dataset('joelito/Multi_Legal_Pile', config, split='train', cache_dir=args.cache_dir, streaming=True)

filtered_list = []
part_number = 1
count = 0
removed = 0
for data in dataset:
    if len(data['text'].split(" ")) <= 1024:
        count += 1
        filtered_list.append(data['text'])
        if len(filtered_list) == 500000:
            df = pd.DataFrame()
            df['text'] = filtered_list
            df.to_csv(f'en_all_filtered_1024_part_{part_number}.tsv', sep='\t', index=False)
            part_number += 1
            filtered_list = []
        print(f'processing part number : {part_number} | count number : {count}')
    else:
        removed += 1
print(f'used count {count} | removed count {removed}')
