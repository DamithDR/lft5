import argparse

import pandas as pd

from data.CaseHoldData import CaseHoldData
from data.LedgarData import LedgarData
from data.MultiEURLEXData import MultiEURLEXData
from data.MultiLexSumData import MultiLexSumData
from data.PrivacyQAData import PrivacyQAData
from data.ScotusData import ScotusData
from data.Uk_INabsData import UK_INabsData
import config.word_limit


def run(args):
    test_files = [
        'Test_casehold.tsv',
        'Test_ledgar.tsv',
        'Test_multi_eurlex.tsv',
        'Test_multi_lexsum.tsv',
        'Test_privacy_qa.tsv',
        'Test_scotus.tsv',
        'Test_uk_abs_in_abs.tsv'
    ]

    results = pd.read_csv('results/finetuned_meta-llama_Llama-2-7b-chat-hfoutputs_1prompt_all_predictions.tsv',
                          sep='\t')
    results_list = results['predictions'].tolist()
    file_map = {
        'Test_casehold.tsv': (CaseHoldData, 'casehold/casehold'),
        'Test_ledgar.tsv': (LedgarData, 'lex_glue'),
        'Test_multi_eurlex.tsv': (MultiEURLEXData, 'multi_eurlex'),
        'Test_privacy_qa.tsv': (PrivacyQAData, 'LegalLLMs/privacy-qa'),
        'Test_uk_abs_in_abs.tsv': (UK_INabsData, 'joelniklaus/legal_case_document_summarization'),
        'Test_multi_lexsum.tsv': (MultiLexSumData, 'allenai/multi_lexsum'),
        'Test_scotus.tsv': (ScotusData, 'lex_glue')
    }

    current_index = 0
    for file in test_files:
        print(f'processing {file}')
        data_class, data_file = file_map.get(file)
        data_class = data_class(data_source=data_file, prompts=[], tokenizer_name=args.tokeniser)
        no_of_instances = data_class.get_no_of_test_instances()

        end_index = current_index + no_of_instances

        print(f'no of instances {no_of_instances}')

        predictions = results_list[current_index:end_index]
        current_index = end_index

        data_class.evaluate_results(predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''preprocess prompts''')
    parser.add_argument('--word_limit', type=int, required=False, help='word_limit')
    parser.add_argument('--tokeniser', type=str, required=True, help='the namer of the tokeniser')
    parser.add_argument('--file_path', type=str, required=True, default='data/permuted_data/1_prompt/',
                        help='file_path')
    args = parser.parse_args()
    if args.word_limit:
        config.word_limit.CONFIG['value'] = args.word_limit
    run(args)
