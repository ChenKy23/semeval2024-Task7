class instr_template:
    def __init__(self):
        self.input_template = {}
        self.label_template = {}

    def load_hg_template(self):
        self.input_template['instr'] = f"""Generate a headline with numbers for following news:
news: {{news}}
headline: """

        self.input_template['icl'] = f"""Generate a headline with numbers for the news, following this example:
example news: {{similar_news}}
example headline: {{similar_headline}}
news: {{news}}
headline: """
 
    def load_hr_template(self):
        self.input_template['opt'] = f"""Calculate some value in following news so that the answer can replace then # in masked headline.
news: {{news}}
masked headline: {{mskh}}"""

        self.input_template['cot'] = f"""According to the news, reason step by step to predict a number to fill in # of masked headline.
news: {{news}}
masked headline: {{mskh}}"""

        self.label_template['opt'] = f"""Calculation: {{calculation}}; Answer: {{ans}}"""

        self.label_template['cot'] = f"""{{cot}} The number is {{ans}}"""