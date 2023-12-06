import pandas as pd

from data.DataClass import DataClass
from data.datafiles.scotus_categories import categories


class ScotusData(DataClass):
    def __init__(self, data_source, prompts):
        super().__init__(data_source, prompts, data_config="scotus", context_alias='{opinion}')
        self.scotus_categories = categories

    def generate_permutations(self):
        permutations = []
        for prompt in self.prompts:
            for opinion, label in zip(self.input_df['text'], self.input_df['label']):
                full_input = self.generate_prompt(prompt=prompt, context=opinion, options='',
                                                  answer=self.scotus_categories[label])
                permutations.append(full_input)

        return pd.DataFrame({'instructions': permutations})
