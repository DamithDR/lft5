import os
from abc import abstractmethod

import pandas as pd
from datasets import load_dataset


class DataClass:
    def __init__(self, data_source, prompts, data_config=None, context_alias="<default>", options_alias='<default>', ):
        self.data_source = data_source
        self.data_config = data_config
        self.prompts = prompts
        self.context_alias = context_alias
        self.option_alias = options_alias
        if os.path.isfile(f'data/datafiles/{self.data_source}'):
            if str(self.data_source).endswith('csv'):
                self.input_df = pd.read_csv(f'data/datafiles/{self.data_source}')
            if str(self.data_source).endswith('tsv'):
                self.input_df = pd.read_csv(f'data/datafiles/{self.data_source}', sep='\t')
            elif str(self.data_source).endswith('jsonl'):
                self.input_df = pd.read_json(f'data/datafiles/{self.data_source}', lines=True)
        else:
            if data_config:
                train_dataset = load_dataset(data_source, data_config, split='train')
                test_dataset = load_dataset(data_source, data_config, split='test')
            else:
                train_dataset = load_dataset(data_source, split='train')
                test_dataset = load_dataset(data_source, split='test')
            self.input_df = train_dataset.to_pandas()
            self.test_input_df = test_dataset.to_pandas()

    def generate_prompt(self, prompt, context, options, answer):
        return f"""
        <human>: {prompt.replace(self.context_alias, context).replace(self.option_alias, options)}
        <assistant>: {answer}
        """.strip()

    def generate_permutations(self):
        train = []
        test = []
        for prompt in self.prompts:
            train_permutations = self.permute(prompt, self.input_df)
            train.extend(train_permutations)
            test_permutations = self.permute(prompt, self.test_input_df, omit_ans=True)
            test.extend(test_permutations)
        return pd.DataFrame({'instructions': train}), pd.DataFrame({'instructions': test})

    @abstractmethod
    def permute(self, prompt, df, omit_ans=False):
        pass