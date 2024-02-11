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


def run(args):
    base_path = args.base_path if args.base_path else 'prompts/raw'
    save_path = args.save_path if args.save_path else 'data/permuted_data/'

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
    for file in prompt_files:
        print(f'processing {file}')
        data_class, data_file = file_map.get(file)
        prompt_data = pd.read_csv(f'{base_path}/{file}', sep='\t')
        data_class = data_class(data_source=data_file, prompts=prompt_data['Prompt'], tokenizer_name=args.tokeniser)
        print(f'generating permutations for {file}')
        train, test = data_class.generate_permutations()
        train.to_csv(f'{save_path}Train_{file}', sep='\t', index=False)
        test.to_csv(f'{save_path}Test_{file}', sep='\t', index=False)
        print(f'finished processing {file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''preprocess prompts''')
    parser.add_argument('--word_limit', type=int, required=False, help='word_limit')
    parser.add_argument('--tokeniser', type=str, required=True, help='the namer of the tokeniser')
    parser.add_argument('--base_path', type=str, required=False, help='base directory path', default='prompts/raw')
    parser.add_argument('--save_path', type=str, required=False, help='save directory path',
                        default='data/permuted_data/')
    args = parser.parse_args()
    if args.word_limit:
        config.word_limit.CONFIG['value'] = args.word_limit
    run(args)
