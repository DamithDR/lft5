import pandas as pd

from data.DataClass import DataClass


class UK_INabsData(DataClass):
    def __init__(self, data_source, prompts):
        super().__init__(data_source, prompts, context_alias='{case}')

    def generate_permutations(self):
        permutations = []
        for prompt in self.prompts:
            for judgement, summary, in zip(self.input_df['judgement'], self.input_df['summary']):
                full_input = self.generate_prompt(prompt=prompt, context=judgement, options='', answer=summary)
                permutations.append(full_input)
        return pd.DataFrame({'instructions': permutations})
