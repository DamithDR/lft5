import math

from data.DataClass import DataClass


class CaseHoldData(DataClass):
    def __init__(self, data_source, prompts):
        super().__init__(data_source=data_source, prompts=prompts, context_alias='{case}', options_alias='{answers}')

    def permute(self, prompt, df, omit_ans=False):
        permutations = []
        for context, ans0, ans1, ans2, ans3, ans4, correct_ans in zip(df['citing_prompt'],
                                                                      df['holding_0'],
                                                                      df['holding_1'],
                                                                      df['holding_2'],
                                                                      df['holding_3'],
                                                                      df['holding_4'],
                                                                      df['label'],
                                                                      ):
            if not math.isnan(int(correct_ans)):
                options = f"""
                    1. {ans0}\n
                    2. {ans1}\n
                    3. {ans2}\n
                    4. {ans3}\n
                    5. {ans4}
                    """
                if omit_ans:
                    answer = ''
                else:
                    ans_list = [ans0, ans1, ans2, ans3, ans4]
                    answer = ans_list[int(correct_ans)]

                full_input = self.generate_prompt(prompt=prompt, context=context, options=options, answer=answer)
                permutations.append(full_input)
        return permutations
