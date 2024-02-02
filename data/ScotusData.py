from data.DataClass import DataClass
from data.datafiles.scotus_categories import categories


class ScotusData(DataClass):

    def __init__(self, data_source, prompts, tokenizer_name):
        super().__init__(data_source, prompts, data_config="scotus", context_alias='{opinion}',
                         tokenizer_name=tokenizer_name)
        self.scotus_categories = categories
        self.filter_dataset()

    def permute(self, prompt, df, omit_ans=False):
        permutations = []
        for opinion, label in zip(df['text'], df['label']):
            if omit_ans:
                answer = ''
            else:
                answer = self.scotus_categories[label]
            full_input = self.generate_prompt(prompt=prompt, context=opinion, options='',
                                              answer=answer)
            permutations.append(full_input)
        return permutations
