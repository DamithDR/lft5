from data.DataClass import DataClass


class PrivacyQAData(DataClass):

    def __init__(self, data_source, prompts):
        super().__init__(data_source, prompts, context_alias='{answer}', options_alias='{question}')

    def permute(self, prompt, df, omit_ans=False):
        permutations = []
        for question, context, label in zip(df['Query'], df['Segment'],df['Label']):

            if omit_ans:
                label = ''
            full_input = self.generate_prompt(prompt=prompt, context=context, options=question, answer=label)
            permutations.append(full_input)
        return permutations
