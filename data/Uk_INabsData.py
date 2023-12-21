from data.DataClass import DataClass


class UK_INabsData(DataClass):

    def __init__(self, data_source, prompts):
        super().__init__(data_source, prompts, context_alias='{case}')
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

        self.input_df['word_count'] = self.input_df['judgement'].apply(lambda x: len(x.split(' ')))
        self.test_input_df['word_count'] = self.test_input_df['judgement'].apply(lambda x: len(x.split(' ')))

        self.input_df = self.input_df.drop(self.input_df[self.input_df['word_count'] > self.word_limit].index)
        self.test_input_df = self.test_input_df.drop(
            self.test_input_df[self.test_input_df['word_count'] > self.word_limit].index)
