import os

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from data.MultiLexSumData import MultiLexSumData
from data.CaseHoldData import CaseHoldData
from data.LedgarData import LedgarData
from data.MultiEURLEXData import MultiEURLEXData
from data.PrivacyQAData import PrivacyQAData

from data.Uk_INabsData import UK_INabsData

base_path = 'prompts/raw'
prompt_files = os.listdir(base_path)

print(prompt_files)

file_map = {
    "CaseHOLD.tsv": (CaseHoldData, 'casehold/casehold'),
    "LEDGAR.tsv": (LedgarData, 'lex_glue'),
    "MultiEURLEX.tsv": (MultiEURLEXData, 'multi_eurlex'),
    "PrivacyQA.tsv": (PrivacyQAData, 'policy_train_data.tsv'),
    "UK-Abs_IN-Abs.tsv": (UK_INabsData, 'joelniklaus/legal_case_document_summarization'),
    "multi_lexsum.tsv": (MultiLexSumData, 'allenai/multi_lexsum')
}

prompt_files=["LEDGAR.tsv"]

def run():
    for file in prompt_files:
        data_class, data_file = file_map.get(file)
        prompt_data = pd.read_csv(f'{base_path}/{file}', sep='\t')
        data_class = data_class(data_source=data_file, prompts=prompt_data['Prompt'])

        data = data_class.generate_permutations()
        data.to_csv(f'data/permuted_data/Data_{file}', sep='\t', index=False)


if __name__ == '__main__':
    run()
