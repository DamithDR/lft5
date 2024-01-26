import pandas as pd
from sklearn.model_selection import train_test_split

files_list = [f'en_all_filtered_1024_part_{i}.tsv' for i in range(1, 18)]
dataframes = [pd.read_csv(f'output_files/{file}', sep='\t') for file in files_list]
df = pd.concat(dataframes, ignore_index=True)
df = df.dropna(how='all')

train, validation = train_test_split(df, test_size=0.1, random_state=777, shuffle=True)

train.to_csv('training_file.tsv', sep='\t', index=False)
validation.to_csv('validation_file.tsv', sep='\t', index=False)
