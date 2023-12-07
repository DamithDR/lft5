from data.DataClass import DataClass


class MultiLexSumData(DataClass):

    def __init__(self, data_source, prompts):
        super().__init__(data_source, prompts, data_config='v20220616', context_alias='{case}')

    def permute(self, prompt, df, omit_ans=False):
        permutations = []

        for judgement, summary in zip(df['sources'][0], df['summary/long']):
            if omit_ans:
                summary = ''
            full_input = self.generate_prompt(prompt=prompt, context=judgement, options='', answer=summary)
            permutations.append(full_input)
        return permutations
