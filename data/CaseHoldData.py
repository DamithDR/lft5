import math

from data.DataClass import DataClass


class CaseHoldData(DataClass):

    def __init__(self, data_source, prompts):
        super().__init__(data_source=data_source, prompts=prompts, context_alias='{case}', options_alias='{answers}')
        self.filter_dataset()

    def permute(self, prompt, df, omit_ans=False):
        permutations = []
        for context, ans0, ans1, ans2, ans3, ans4, correct_ans in zip(df['citing_prompt'],
                                                                      df['holding_0'],
                                                                      df['holding_1'],
                                                                      df['holding_2'],
                                                                      df['holding_3'],
                                                                      df['holding_4'],
                                                                      df['label'],
                                                                      ):
            if not math.isnan(int(correct_ans)):
                options = f"""1. {ans0.strip()}\n2. {ans1.strip()}\n3. {ans2.strip()}\n4. {ans3.strip()}\n5. {ans4.strip()}"""
                if omit_ans:
                    answer = ''
                else:
                    ans_list = [ans0.strip(), ans1.strip(), ans2.strip(), ans3.strip(), ans4.strip()]
                    answer = ans_list[int(correct_ans)]

                full_input = self.generate_prompt(prompt=prompt, context=context, options=options, answer=answer)
                permutations.append(full_input)
        return permutations

    def filter_dataset(self):
        self.flattern_text()
        self.input_df['word_count'] = self.input_df['concatenated'].apply(lambda x: len(x.split(' ')))
        self.test_input_df['word_count'] = self.test_input_df['concatenated'].apply(lambda x: len(x.split(' ')))

        self.input_df = self.input_df.drop(self.input_df[self.input_df['word_count'] > self.word_limit].index)
        self.test_input_df = self.test_input_df.drop(
            self.test_input_df[self.test_input_df['word_count'] > self.word_limit].index)

    def flattern_text(self):
        self.input_df['concatenated'] = self.input_df['citing_prompt'] + self.input_df['holding_0'] \
                                        + self.input_df['holding_1'] + self.input_df['holding_2'] + self.input_df[
                                            'holding_3'] + self.input_df['holding_4'] + \
                                        self.input_df['label']

        self.test_input_df['concatenated'] = self.test_input_df['citing_prompt'] + self.test_input_df['holding_0'] + \
                                             self.test_input_df[
                                                 'holding_1'] + self.test_input_df['holding_2'] + self.test_input_df[
                                                 'holding_3'] + self.test_input_df['holding_4'] + \
                                             self.test_input_df['label']
