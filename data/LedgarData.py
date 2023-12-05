import pandas as pd

from data.DataClass import DataClass


class LedgarData(DataClass):
    def __init__(self, data_source, prompts):
        super().__init__(data_source, prompts, context_alias='<provision>')

    def generate_permutations(self):
        permutations = []
        for prompt in self.prompts:
            for provision, label in zip(self.input_df['provision'], self.input_df['label']):
                full_input = self.generate_prompt(prompt=prompt, context=provision, options='', answer=",".join(label))
                permutations.append(full_input)

        return pd.DataFrame({'instructions': permutations})
