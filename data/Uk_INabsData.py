from data.DataClass import DataClass


class UK_INabsData(DataClass):

    def __init__(self, data_source, prompts):
        super().__init__(data_source, prompts, context_alias='{case}')

    def permute(self, prompt, df, omit_ans=False):
        permutations = []
        for judgement, summary, in zip(df['judgement'], df['summary']):
            if omit_ans:
                summary = ''
            full_input = self.generate_prompt(prompt=prompt, context=judgement, options='', answer=summary)
            permutations.append(full_input)
        return permutations
