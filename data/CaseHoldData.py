import math
import os
import pandas as pd

from data.DataClass import DataClass


class CaseHoldData(DataClass):
    def __init__(self, data_source, prompts):
        super().__init__(data_source=data_source, prompts=prompts, context_alias='{case}', options_alias='{answers}')

    def generate_permutations(self):
        permutations = []
        for prompt in self.prompts:
            for context, ans0, ans1, ans2, ans3, ans4, correct_ans in zip(self.input_df['1'],
                                                                          self.input_df['2'],
                                                                          self.input_df['3'],
                                                                          self.input_df['4'],
                                                                          self.input_df['5'],
                                                                          self.input_df['6'],
                                                                          self.input_df['12'],
                                                                          ):
                # if not math.isnan(ans0) and not math.isnan(ans1) and not math.isnan(ans2) and not math.isnan(ans3) and not math.isnan(ans4) and
                if not math.isnan(correct_ans):
                    options = f"""
                        1. {ans0}\n
                        2. {ans1}\n
                        3. {ans2}\n
                        4. {ans3}\n
                        5. {ans4}
                        """
                    ans_list = [ans0, ans1, ans2, ans3, ans4]
                    answer = ans_list[int(correct_ans)]
                    full_input = self.generate_prompt(prompt=prompt, context=context, options=options, answer=answer)
                    permutations.append(full_input)
        return pd.DataFrame({'instructions': permutations})
