import argparse
import os

import pandas as pd

import config.word_limit
from data.MultiLexSumData import MultiLexSumData
from data.CaseHoldData import CaseHoldData
from data.LedgarData import LedgarData
from data.MultiEURLEXData import MultiEURLEXData
from data.PrivacyQAData import PrivacyQAData
from data.ScotusData import ScotusData

from data.Uk_INabsData import UK_INabsData

base_path = 'prompts/raw'
prompt_files = os.listdir(base_path)

print(prompt_files)

file_map = {
    "casehold.tsv": (CaseHoldData, 'casehold/casehold'),
    "ledgar.tsv": (LedgarData, 'lex_glue'),
    "multi_eurlex.tsv": (MultiEURLEXData, 'multi_eurlex'),
    "privacy_qa.tsv": (PrivacyQAData, 'LegalLLMs/privacy-qa'),
    "uk_abs_in_abs.tsv": (UK_INabsData, 'joelniklaus/legal_case_document_summarization'),
    "multi_lexsum.tsv": (MultiLexSumData, 'allenai/multi_lexsum'),
    "scotus.tsv": (ScotusData, 'lex_glue')
}


def run():
    for file in prompt_files:
        data_class, data_file = file_map.get(file)
        prompt_data = pd.read_csv(f'{base_path}/{file}', sep='\t')
        data_class = data_class(data_source=data_file, prompts=prompt_data['Prompt'])

        train, test = data_class.generate_permutations()
        train.to_csv(f'data/permuted_data/Train_{file}', sep='\t', index=False)
        test.to_csv(f'data/permuted_data/Test_{file}', sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''preprocess prompts''')
    parser.add_argument('--word_limit', type=int, required=False, help='word_limit')
    args = parser.parse_args()
    if args.word_limit:
        config.word_limit.CONFIG['value'] = args.word_limit
    run()
