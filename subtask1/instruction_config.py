class instr_template:
    def __init__(self):
        self.input_template = {}
        self.label_template = {}

    def load_qp_template(self):
        self.input_template['instr_comment'] = f"""Predict the magnitude for [Num] in the comment: {{masked}}
magnitude: """

        self.input_template['instr_headline'] = f"""Predict the magnitude for [Num] in the headline: {{masked}}
magnitude: """
        
        self.input_template['icl_comment'] = f"""Predict the magnitude for [Num] in the comment: COSTCO WHOLESALE <COST.O> UP [Num] PCT IN PREMARKET TRADING AFTER  RESULTS
magnitude: 0
Predict the magnitude for [Num] in the comment: BRIEF-Henan Shuanghui Investment's H1 net profit up [Num] pct
magnitude: 1
Predict the magnitude for [Num] in the comment: Enbridge Energy to sell [Num] pct interest in midstream business
magnitude: 2
Predict the magnitude for [Num] in the comment: EASTERN TREADS LTD - SEPT QTR NET SALES 217 MLN RUPEES VS [Num] MLN  RUPEES YR AGO
magnitude: 3
Predict the magnitude for [Num] in the comment: NSEI Block Deal Ultratech Cement 50400 shares at [Num]0 INR <ULTC.NS>
magnitude: 4
Predict the magnitude for [Num] in the comment: NYSE ORDER IMBALANCE <SWN.N> [Num] SHARES ON BUY SIDE
magnitude: 5
Predict the magnitude for [Num] in the comment: NYSE INDICATION <BRKa.N> LAST 132295.0 BID 131000.0 ASK [Num]
magnitude: 6
Predict the magnitude for [Num] in the comment: This is a TEST snap for ECONTEST2=ECI ([Num]), Please ignore.
magnitude: 7
Now complete the following example.
Predict the magnitude for [Num] in the comment: {{masked}}
magnitude: """
        
        self.input_template['icl_headline'] = f"""Predict the magnitude for [Num] in the headline: America in command of Interliga Group A after 3-[Num] win over Atlante
magnitude: 0
Predict the magnitude for [Num] in the headline: Apple's App Store surpasses [Num] billion downloads
magnitude: 1
Predict the magnitude for [Num] in the headline: Apple iPad tablet computer expected to be unveiled Jan 26 or [Num]
magnitude: 2
Predict the magnitude for [Num] in the headline: Are we failures if we don't achieve goals [Num]%? : putting goals and achievements into perspective
magnitude: 3
Predict the magnitude for [Num] in the headline: 100 Most Anticipated Books releasing in [Num]
magnitude: 4
Predict the magnitude for [Num] in the headline: San Diego principal pleaded guilty to stealing $[Num] from Oceanside school district
magnitude: 5
Predict the magnitude for [Num] in the headline: Wow! See The $[Num] Necklace That 'The Bachelor' Will Give To His Mystery Date Tonight
magnitude: 6
Predict the magnitude for [Num] in the headline: [Num] iTunes downloads and counting
magnitude: 7
Now complete the following example.
Predict the magnitude for [Num] in the headline: {{masked}}
magnitude: """
        
    def load_qnli_template(self):
        self.input_template['icl'] = f"""Based on the options, predict the relationship between the following two statements.
Statement1: Bob has 3 New Year cards
Statement2: Bob's New Year card is less than 2.
Answer: contradiction
Based on the options, predict the relationship between the following two statements.
Statement1: Jack and Christina are less than 340 feet apart.
Statement2: Jack and Christina's home is 240 feet apart
Answer: neutral
Based on the options, predict the relationship between the following two statements.
If Bob's monthly income increases by 14 %, he will earn $ 678
If Bob's monthly income increases by less than 74 %, he will earn $678
Answer: entailment
Now complete following example.
Based on the options, predict the relationship between the following two statements.
Statement1: {{statement1}}
Statement2: {{statement2}}
Options: {{options}}"""

        self.input_template['instr'] = f"""Based on the options, predict the relationship between the following two statements.
Statement1: {{statement1}}
Statement2: {{statement2}}
Options: {{options}}"""