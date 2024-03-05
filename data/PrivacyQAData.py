import pandas as pd

from data.DataClass import DataClass


class PrivacyQAData(DataClass):

    def __init__(self, data_source, prompts, tokenizer_name):
        super().__init__(data_source, prompts, context_alias='{answer}', options_alias='{question}',
                         tokenizer_name=tokenizer_name)
        self.filter_dataset()

    def permute(self, prompt, df, omit_ans=False):
        permutations = []
        for question, context, label in zip(df['Query'], df['Segment'], df['Label']):

            if omit_ans:
                label = ''
            full_input = self.generate_prompt(prompt=prompt, context=context, options=question, answer=label)
            permutations.append(full_input)
        return permutations

    def filter_dataset(self):
        print(f'filter dataset started {self.data_source}')
        if self.tokenizer_name is not None:
            if len(self.input_df) > 0:
                data = self.input_df['Segment'].tolist()
                tokens = self.tokenizer(data)
                lengths = [len(token_lst) for token_lst in tokens['input_ids']]
                self.input_df['tokens'] = lengths
                self.input_df = self.input_df.drop(self.input_df[self.input_df['tokens'] > self.word_limit].index)
            if len(self.test_input_df) > 0:
                data = self.test_input_df['Segment'].tolist()
                tokens = self.tokenizer(data)
                lengths = [len(token_lst) for token_lst in tokens['input_ids']]
                self.test_input_df['tokens'] = lengths
                self.test_input_df = self.test_input_df.drop(
                    self.test_input_df[self.test_input_df['tokens'] > self.word_limit].index)
        print(f'filter dataset finished {self.data_source}')

    def filter_dataset_whitespace(self):

        self.input_df['word_count'] = self.input_df['Segment'].apply(lambda x: len(x.split(' ')))
        self.test_input_df['word_count'] = self.test_input_df['Segment'].apply(lambda x: len(x.split(' ')))

        self.input_df = self.input_df.drop(self.input_df[self.input_df['word_count'] > self.word_limit].index)
        self.test_input_df = self.test_input_df.drop(
            self.test_input_df[self.test_input_df['word_count'] > self.word_limit].index)

    def get_gold_standards(self):
        return self.test_input_df['Label'].tolist()

    def evaluate_results(self, predictions):
        df = pd.DataFrame({'predictions': predictions})
        df.to_csv('privacyqa_predictions.tsv', sep='\t', index=False)

        print(f'privacyqa predictions size = {len(predictions)}')

