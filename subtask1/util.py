import json
import numpy as np
import torch
import random, os

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def remove_key_json(json_data, key_to_remove):
    return [{key: value for key, value in data.items() if key not in key_to_remove} for data in json_data]

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

def trans_to_dict_qp(data):
    data = remove_key_json(data, ['length', 'offset', 'comment_sci_10E_char', 'comment_sci_10E', 'title_sci_10E_char', 'title_sci_10E', 'UNIQUE_STORY_INDEX'])

    keys = data[0].keys()
    data_dic = {}

    for key in keys:
        data_dic[key] = []

    for item in data:
        for key in keys:
            val = item[key]
            if key == "number":
                val = float(val)
            elif key == 'magnitude' or key == 'offset' or key == 'id' or key == 'length':
                val = int(val)
            else:
                val = str(val)
            data_dic[key].append(val)

    return data_dic

def trans_to_dict_qnli(data):
    data = remove_key_json(data, ['type', 'statement2_sci_10E_char', 'statement2_mask', 'EQUATE', 'statement1_mask', 'statement1_sci_10E', 'statement1_sci_10E_char', 'statement2_sci_10E'])

    keys = data[0].keys()
    data_dic = {}

    for key in keys:
        data_dic[key] = []

    for item in data:
        for key in keys:
            sstr = item[key]

            sstr = str(sstr)
            
            data_dic[key].append(sstr)
    
    return data_dic