import pandas as pd

from data.DataClass import DataClass
from data.datafiles.ledgar_categories import LEDGAR_CATEGORIES


class LedgarData(DataClass):
    def __init__(self, data_source, prompts):
        super().__init__(data_source, prompts,data_config="ledgar", context_alias='<provision>')
        self.ledgar_categories = LEDGAR_CATEGORIES

    def generate_permutations(self):
        permutations = []
        for prompt in self.prompts:
            for provision, label in zip(self.input_df['text'], self.input_df['label']):
                full_input = self.generate_prompt(prompt=prompt, context=provision, options='', answer=self.ledgar_categories[label])
                permutations.append(full_input)

        return pd.DataFrame({'instructions': permutations})
