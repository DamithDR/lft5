from datasets import load_dataset

config = 'en_all'
dataset = load_dataset('joelito/Multi_Legal_Pile', config, split='train', streaming=True)

df = dataset.to_pandas()

df['word_count'] = df['text'].apply(lambda x: len(x.split(' ')))

df = df.drop(df[df['word_count'] > 1024].index)

df.to_csv('en_all_filtered.tsv', sep='\t', index=False)
