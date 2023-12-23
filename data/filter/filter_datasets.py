import argparse

from datasets import load_dataset

parser = argparse.ArgumentParser(description='''convert dataset''')
parser.add_argument('--cache_dir', type=str, required=False, help='model_name')
args = parser.parse_args()

config = 'en_all'
if args.cache_dir:
    dataset = load_dataset('joelito/Multi_Legal_Pile', config, split='train', cache_dir=args.cache_dir)
else:
    dataset = load_dataset('joelito/Multi_Legal_Pile', config, split='train')
df = dataset.to_pandas()

df['word_count'] = df['text'].apply(lambda x: len(x.split(' ')))
df = df.drop(df[df['word_count'] > 1024].index)
df.to_csv('en_all_filtered_1024.tsv', sep='\t', index=False)
