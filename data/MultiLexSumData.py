from data.DataClass import DataClass


class MultiLexSumData(DataClass):

    def __init__(self, data_source, prompts):
        super().__init__(data_source, prompts, data_config='v20220616', context_alias='{case}')
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
