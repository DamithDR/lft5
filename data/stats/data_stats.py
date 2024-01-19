import json

import pandas as pd

data_files = [
    'Test_casehold.tsv',
    'Test_ledgar.tsv',
    'Test_multi_eurlex.tsv',
    'Test_multi_lexsum.tsv',
    'Test_privacy_qa.tsv',
    'Test_scotus.tsv',
    'Test_uk_abs_in_abs.tsv',
    'Train_casehold.tsv',
    'Train_ledgar.tsv',
    'Train_multi_eurlex.tsv',
    'Train_multi_lexsum.tsv',
    'Train_privacy_qa.tsv',
    'Train_scotus.tsv',
    'Train_uk_abs_in_abs.tsv'
]

final_results = dict()

word_count_list = []
for data_file in data_files:
    counts_dict = {
        "< 512": 0,
        "512 - 1024": 0,
        "1024 - 2048": 0,
        "2048 - 4096": 0,
        "4096 - 8192": 0,
        "8192 - 16384": 0,
        "16384 - 32768": 0,
        "32768 - 65536": 0,
        "65536 <": 0
    }
    d_set = pd.read_csv(f'data/permuted_data/{data_file}', sep='\t')
    # d_set = pd.read_csv(f'D:/DataSets/legalfinetune/{data_file}', sep='\t')
    for row in d_set['instructions']:
        no_of_words = len(row.split(' '))
        word_count_list.append(no_of_words)
        if no_of_words <= 512:
            counts_dict["< 512"] += 1
        elif 512 < no_of_words <= 1024:
            counts_dict["512 - 1024"] += 1
        elif 1024 < no_of_words <= 2048:
            counts_dict["1024 - 2048"] += 1
            # print(f'{no_of_words} > {data_file}')
            print(no_of_words)
            # if no_of_words > 1100:
            #     print(row)
        elif 2048 < no_of_words <= 4096:
            counts_dict["2048 - 4096"] += 1
        elif 4096 < no_of_words <= 8192:
            counts_dict["4096 - 8192"] += 1
        elif 8192 < no_of_words <= 16384:
            counts_dict["8192 - 16384"] += 1
        elif 16384 < no_of_words <= 32768:
            counts_dict["16384 - 32768"] += 1
        elif 32768 < no_of_words <= 65536:
            counts_dict["32768 - 65536"] += 1
        else:
            counts_dict["65536 <"] += 1

    final_results[data_file] = counts_dict
    print(final_results[data_file])
counts = sorted(word_count_list,reverse=True)
with open("filtered_file_wise_counts.json", "w") as file:
    json.dump(final_results, file)
