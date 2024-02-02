from data.DataClass import DataClass


class UK_INabsData(DataClass):

    def __init__(self, data_source, prompts, tokenizer_name):
        super().__init__(data_source, prompts, context_alias='{case}', tokenizer_name=tokenizer_name)
        self.filter_dataset()

    def permute(self, prompt, df, omit_ans=False):
        permutations = []
        for judgement, summary, in zip(df['judgement'], df['summary']):
            if omit_ans:
                summary = ''
            full_input = self.generate_prompt(prompt=prompt, context=judgement, options='', answer=summary)
            permutations.append(full_input)
        return permutations

    def filter_dataset(self):
        if self.tokenizer_name is not None:
            if len(self.input_df) > 0:
                concat = self.input_df['judgement'] + self.input_df['summary']
                data = concat.tolist()
                tokens = self.tokenizer(data)
                lengths = [len(token_lst) for token_lst in tokens['input_ids']]
                self.input_df['tokens'] = lengths
                self.input_df = self.input_df.drop(self.input_df[self.input_df['tokens'] > self.word_limit].index)
            if len(self.test_input_df) > 0:
                concat_test = self.test_input_df['judgement'] + self.test_input_df['summary']
                data = concat_test.tolist()
                tokens = self.tokenizer(data)
                lengths = [len(token_lst) for token_lst in tokens['input_ids']]
                self.test_input_df['tokens'] = lengths
                self.test_input_df = self.test_input_df.drop(
                    self.test_input_df[self.test_input_df['tokens'] > self.word_limit].index)

    def filter_dataset_whitespace(self):
        concat = self.input_df['judgement'] + self.input_df['summary']
        self.input_df['word_count'] = concat.apply(lambda x: len(x.split(' ')))
        concat_test = self.test_input_df['judgement'] + self.test_input_df['summary']
        self.test_input_df['word_count'] = concat_test.apply(lambda x: len(x.split(' ')))

        self.input_df = self.input_df.drop(self.input_df[self.input_df['word_count'] > self.word_limit].index)
        self.test_input_df = self.test_input_df.drop(
            self.test_input_df[self.test_input_df['word_count'] > self.word_limit].index)
