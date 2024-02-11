from data.DataClass import DataClass


class MultiLexSumData(DataClass):

    def __init__(self, data_source, prompts, tokenizer_name):
        super().__init__(data_source, prompts, data_config='v20220616', context_alias='{case}',
                         tokenizer_name=tokenizer_name)
        self.filter_dataset()

    def permute(self, prompt, df, omit_ans=False):
        permutations = []

        for documents, summary in zip(df['sources'], df['summary/long']):
            text = "-".join(documents.tolist())
            if omit_ans:
                summary = ''
            full_input = self.generate_prompt(prompt=prompt, context=text, options='', answer=summary)
            permutations.append(full_input)
        return permutations

    def filter_dataset(self):
        print(f'filter dataset started {self.data_source}')
        self.flattern_text()
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
        self.flattern_text()

        self.input_df['word_count'] = self.input_df['text'].apply(lambda x: len(x.split(' ')))
        self.test_input_df['word_count'] = self.test_input_df['text'].apply(lambda x: len(x.split(' ')))

        self.input_df = self.input_df.drop(self.input_df[self.input_df['word_count'] > self.word_limit].index)
        self.test_input_df = self.test_input_df.drop(
            self.test_input_df[self.test_input_df['word_count'] > self.word_limit].index)

    def flattern_text(self):
        texts = []
        for documents in self.input_df['sources']:
            text = "-".join(documents.tolist())
            texts.append(text)
        self.input_df['text'] = texts

        texts = []
        for documents in self.test_input_df['sources']:
            text = "-".join(documents.tolist())
            texts.append(text)
        self.test_input_df['text'] = texts
