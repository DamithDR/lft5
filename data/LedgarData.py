from data.DataClass import DataClass
from data.datafiles.ledgar_categories import LEDGAR_CATEGORIES


class LedgarData(DataClass):
    def __init__(self, data_source, prompts, tokenizer):
        super().__init__(data_source, prompts, data_config="ledgar", context_alias='<provision>',
                         tokenizer_name=tokenizer)
        self.ledgar_categories = LEDGAR_CATEGORIES
        self.filter_dataset()

    def permute(self, prompt, df, omit_ans=False):
        permutations = []
        for provision, label in zip(df['text'], df['label']):
            if omit_ans:
                answer = ''
            else:
                answer = self.ledgar_categories[label]
            full_input = self.generate_prompt(prompt=prompt, context=provision, options='',
                                              answer=answer)
            permutations.append(full_input)
        return permutations
