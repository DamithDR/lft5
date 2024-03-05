import os
from abc import abstractmethod

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

import config.word_limit


class DataClass:
    def __init__(self, data_source, prompts=None, data_config=None, context_alias="<default>",
                 options_alias='<default>',
                 tokenizer_name=None):
        self.data_source = data_source
        self.data_config = data_config
        self.prompts = prompts
        self.context_alias = context_alias
        self.option_alias = options_alias
        self.tokenizer_name = tokenizer_name
        self.word_limit = config.word_limit.CONFIG['value']
        if self.tokenizer_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
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

    def filter_dataset(self):
        print(f'filter dataset started {self.data_source}')
        if self.tokenizer_name is not None:
            if len(self.input_df) > 0:
                data = self.input_df['text'].tolist()
                tokens = self.tokenizer(data)
                lengths = [len(token_lst) for token_lst in tokens['input_ids']]
                self.input_df['tokens'] = lengths
                self.input_df = self.input_df.drop(self.input_df[self.input_df['tokens'] > self.word_limit].index)
            if len(self.test_input_df) > 0:
                data = self.test_input_df['text'].tolist()
                tokens = self.tokenizer(data)
                lengths = [len(token_lst) for token_lst in tokens['input_ids']]
                self.test_input_df['tokens'] = lengths
                self.test_input_df = self.test_input_df.drop(
                    self.test_input_df[self.test_input_df['tokens'] > self.word_limit].index)
        print(f'filter dataset finished {self.data_source}')

    def filter_dataset_whitespace(self):

        self.input_df['word_count'] = self.input_df['text'].apply(lambda x: len(x.split(' ')))
        self.test_input_df['word_count'] = self.test_input_df['text'].apply(lambda x: len(x.split(' ')))

        self.input_df = self.input_df.drop(self.input_df[self.input_df['word_count'] > self.word_limit].index)
        self.test_input_df = self.test_input_df.drop(
            self.test_input_df[self.test_input_df['word_count'] > self.word_limit].index)

    def generate_prompt(self, prompt, context, options, answer):
        return f"""
        <human>: {prompt.replace(self.context_alias, context).replace(self.option_alias, options)}
        <assistant>: {answer}
        """.strip()

    def generate_permutations(self):
        train = []
        test = []
        test_answers = []
        for prompt in self.prompts:
            train_permutations, _ = self.permute(prompt, self.input_df)
            train.extend(train_permutations)
            test_permutations, answers = self.permute(prompt, self.test_input_df, omit_ans=True)
            test.extend(test_permutations)
            test_answers.extend(answers)
        return pd.DataFrame({'instructions': train}), pd.DataFrame({'instructions': test, 'answers': test_answers})

    @abstractmethod
    def permute(self, prompt, df, omit_ans=False):
        pass

    @abstractmethod
    def evaluate_results(self, predictions):
        pass

    def get_no_of_test_instances(self):
        return len(self.test_input_df)
