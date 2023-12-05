import pandas as pd

from data.DataClass import DataClass


class PrivacyQAData(DataClass):
    def __init__(self, data_source, prompts):
        super().__init__(data_source, prompts, context_alias='{answer}', options_alias='{question}')

    def generate_permutations(self):
        permutations = []
        for prompt in self.prompts:
            for question, context, label in zip(self.input_df['Query'], self.input_df['Segment'],
                                                self.input_df['Label']):
                full_input = self.generate_prompt(prompt=prompt, context=context, options=question, answer=label)
                permutations.append(full_input)
        return pd.DataFrame({'instructions': permutations})
