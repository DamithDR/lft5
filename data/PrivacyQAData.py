from data.DataClass import DataClass


class PrivacyQAData(DataClass):

    def __init__(self, data_source, prompts):
        super().__init__(data_source, prompts, context_alias='{answer}', options_alias='{question}')
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

        self.input_df['word_count'] = self.input_df['Segment'].apply(lambda x: len(x.split(' ')))
        self.test_input_df['word_count'] = self.test_input_df['Segment'].apply(lambda x: len(x.split(' ')))

        self.input_df = self.input_df.drop(self.input_df[self.input_df['word_count'] > self.word_limit].index)
        self.test_input_df = self.test_input_df.drop(
            self.test_input_df[self.test_input_df['word_count'] > self.word_limit].index)
