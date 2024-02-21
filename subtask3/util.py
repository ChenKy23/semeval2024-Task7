import random, json, re
import numpy as np
import torch
import os

from nltk.tokenize import sent_tokenize
from transformers.trainer_callback import TrainerCallback
from sklearn.metrics import accuracy_score

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # torch.set_deterministic(True)
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def transToDict(data):
    keys = data[0].keys()
    data_dic = {}

    for key in keys:
        data_dic[key] = []

    for item in data:
        for key in keys:
            str = item[key]
            if key == 'news':
                str = re.sub(r'\([^)]*\)', '', str, 1).strip()    
            data_dic[key].append(str)
        
    return data_dic

ANS_RE_COT = re.compile(r"The number is (\d{1,3}(?:,\d{3})+|\d+[\/\.]{0,1}\d+|\d+)")
ANS_RE_OPT = re.compile(r"Answer: (\d{1,3}(?:,\d{3})+|\d+[\/\.]{0,1}\d+|\d+)")

INVALID_ANS = "[invalid]"
def extract_answer(completion, type='cot'):
    if type == 'cot':
        match = ANS_RE_COT.search(completion)
    else:
        match = ANS_RE_OPT.search(completion)
    if match:
        match_str = match.group(1).strip()
        return match_str
    else:
        return INVALID_ANS

def batch_find_ans(decoded_list, type='cot'):
    ans_list = []
    for decode in decoded_list:
        ans = extract_answer(decode, type)
        ans_list.append(ans)
    return ans_list

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

def dropout_redundant_hg(data):
    new_data = dict()
    new_data['news'] = []
    new_data['headline'] = []
    new_data['similar_news'] = []
    new_data['similar_headline'] = []
    for item in data:
        news =  item['news']
        news = re.sub(r'\([^)]*\)', '', news, 1).strip()    
        new_data['news'].append(news)
        new_data['headline'].append(item['headline'])
        new_data['similar_news'].append(item['similar_news'])
        new_data['similar_headline'].append(item['similar_headline'])

    return new_data

def dropout_redundant_hr(data, args=None):
    new_data = dict()
    new_data['news'] = []
    new_data['masked headline'] = []
    new_data['calculation'] = []
    new_data['ans'] = []
    if args and args.task == "train":
        new_data['generate_template'] = []
    
    for item in data:
        news =  item['news']
        mskh = item['masked headline']
        cal = item['calculation']
        ans = item['ans']
        if args and args.task == "train":
            cot = item['generate_template']

        if not isinstance(news, str):
           news = str(news) 
        if not isinstance(mskh, str):
            mskh = str(mskh)
        if not isinstance(cal, str):
            cal = str(cal)
        if not isinstance(ans, str):
            ans = str(ans) 
            mskh = replace_empty(mskh)

        new_data['news'].append(news.strip())
        new_data['masked headline'].append(mskh.strip())
        new_data['calculation'].append(cal.strip())
        new_data['ans'].append(ans.strip())

        if args and args.task == "train":
            new_data['generate_template'].append(cot.strip())
    
    return new_data

def replace_empty(input_str):
    output_str = re.sub(r'(_+)', r' # ', input_str)
    output_str = re.sub(r'\s+', ' ', output_str)
    return output_str.strip()

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]
    
def cal_num_acc(pred, num_gt_path, num_type_path):
    pred_all, gt_all = [], []
    pred_copy, gt_copy = [], []
    pred_cal, gt_cal = [], []
    pattern = re.compile(r'\d{1,3}(?:,\d{3})+|\d+[\/\.]{0,1}\d+|\d+')

    with open(num_gt_path) as target, open(num_type_path) as num_type:
        total_num, count_copy, count_cal = 0, 0, 0
        for (hyp, gt, num) in zip(pred, target, num_type):
            gt_split = gt.split("#")
            num_split = num.split("#")

            for sub_gt, sub_num in zip(gt_split, num_split):
                generated_num_list = pattern.findall(hyp)
                
                if(str(sub_num).strip()=='0'): # Copy
                    count_copy += 1
                    if(str(sub_gt).strip() in generated_num_list):
                        pred_copy.append(1)
                        pred_all.append(1)
                    else:
                        pred_copy.append(0)
                        pred_all.append(0)
                    gt_copy.append(1)
                    gt_all.append(1)
                else:
                    count_cal += 1
                    if(str(sub_gt).strip() in generated_num_list):
                        pred_cal.append(1)
                        pred_all.append(1)
                    else:
                        pred_cal.append(0)
                        pred_all.append(0)
                    gt_cal.append(1)
                    gt_all.append(1)
                total_num += 1
        print("All Accuracy: %.6f, Copy Accuracy: %.6f, Cal Accuracy: %.6f"%(accuracy_score(gt_all, pred_all), accuracy_score(gt_copy, pred_copy), accuracy_score(gt_cal, pred_cal)))