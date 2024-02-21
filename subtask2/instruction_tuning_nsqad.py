from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

import json

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import accuracy_score
import random, json, re
from random import randrange, sample
import argparse
import os

import evaluate
import numpy as np
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def remove_duplicate_dicts(input_list):
    result = []
    for item in input_list:
        if item not in result:
            result.append(item)
    return result

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

def transToDict(data):
    data = remove_key_json(data, ['sentences_containing_the_numeral_in_answer_options'])

    keys = data[0].keys()
    data_dic = {}

    for key in keys:
        data_dic[key] = []

    for item in data:
        news = " ".join(item['news_article']).strip()
        data_dic['news_article'].append(re.sub(r'\s+', ' ', news))
        question = item['question_stem'].replace("___",'[Num]').strip()
        data_dic['question_stem'].append(question)
        for ss in item['answer_options']:
            ss = str(ss)
        data_dic['answer_options'].append(item['answer_options'])
        data_dic['target_num'].append(str(item['target_num']).strip())
        data_dic['ans'].append(str(item['ans']).strip())
    
    return data_dic

class instr_template:
    def __init__(self):
        self.input_template = {}
        self.label_template = {}

    def load_nsqad_template(self):
        self.input_template['icl_ch'] = f"""根据新闻，为以下问题中的[Num]选择正确选项
新闻: {{news_article}}
问题: {{question}}
选项:
A {{option1}}
B {{option2}}
C {{option3}} 
D {{option4}}"""
        
        self.input_template['icl_en'] = f"""According to the news, Choose the correct option for [Num] in the following questions.
news: {{news_article}}
question: {{question}}
option:
A {{option1}}
B {{option2}}
C {{option3}} 
D {{option4}}"""

        self.label_template['ans'] = f"""{{ans}} {{num}}"""

def train_and_evaluate(args, tokenizer, tokenized_dataset):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        count_equal_ans = sum(decode_pred.split(" ")[-1].strip() == decoded_label.split(" ")[-1].strip() for decode_pred, decoded_label in zip(decoded_preds, decoded_labels))
        num_acc = round(count_equal_ans/len(decoded_labels)*100, 4)

        preds_list = []
        labels_list = []
        for pred, label in zip(decoded_preds, decoded_labels):
            if pred.startswith("A"):
                preds_list.append(0)
            elif pred.startswith("B"):
                preds_list.append(1)
            elif pred.startswith("C"):
                preds_list.append(2)
            else:
                preds_list.append(3)

            if label.startswith("A"):
                labels_list.append(0)
            elif label.startswith("B"):
                labels_list.append(1)
            elif label.startswith("C"):
                labels_list.append(2)
            elif label.startswith("D"):
                labels_list.append(3)

        opt_acc = accuracy_score(y_pred=preds_list, y_true=labels_list, normalize=True, sample_weight=None)
        macro_f1 = f1_metric.compute(predictions=preds_list, references=labels_list, average="macro")
        micro_f1 = f1_metric.compute(predictions=preds_list, references=labels_list, average="micro")

        result = {}
        result['micro_f1'] = micro_f1['f1']
        result['macro_f1'] = macro_f1['f1']
        result['opt_acc'] = opt_acc
        result['num_acc'] = num_acc

        return result

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    f1_metric = evaluate.load("./f1.py")

    label_pad_token_id = -100

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
        padding='longest'
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_model_path,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        fp16=False,
        learning_rate=args.lr,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warm_up_radio,
        weight_decay=args.weight_decay,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed = args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["dev"],
        compute_metrics=compute_metrics
    )

    torch.cuda.empty_cache()
    # Start training
    trainer.train()

def predict_and_save_res(args, tokenizer=None, tokenized_dataset=None, dataset_test=None):
    def get_predict(model, tokenized_dataset, batch_size = 4, max_new_tokens = 128, sample_set = 'test', device = 'cuda'):
        """
        Get the predictions from the trained model.
        """
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            attention_mask = [torch.tensor(example['attention_mask']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=tokenizer.pad_token_id)
            return input_ids, attention_mask
        
        dataloader = DataLoader(tokenized_dataset[sample_set], batch_size=batch_size, collate_fn=collate_fn)
        model.to(device)
        print('Model loaded to: ', device)
        preds_opt = []
        preds_num = []
        preds_out = []

        for inputs, attention_mask in tqdm(dataloader):
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)

            output_ids = model.generate(input_ids=inputs, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
            
            decode_pred_ans = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for decode_pred in decode_pred_ans:
                num = decode_pred.split(" ")[-1].strip()
                preds_num.append(num)
                if decode_pred.startswith("A"):
                    preds_opt.append(0)
                elif decode_pred.startswith("B"):
                    preds_opt.append(1)
                elif decode_pred.startswith("C"):
                    preds_opt.append(2)
                elif decode_pred.startswith("D"):
                    preds_opt.append(3)

                preds_out.append(decode_pred)

        return preds_opt, preds_num, preds_out
    
    f1_metric = evaluate.load("./f1.py")

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)

    preds_opt, preds_num, preds_out = get_predict(model=model, tokenized_dataset = tokenized_dataset, batch_size=30, max_new_tokens=25, sample_set='test', device = 'cuda')

    labels_opt = [int(sample["ans"]) for sample in dataset_test]
    labels_num = [sample["target_num"] for sample in dataset_test]

    macro_f1 = f1_metric.compute(predictions=preds_opt, references=labels_opt, average="macro")
    micro_f1 = f1_metric.compute(predictions=preds_opt, references=labels_opt, average="micro")

    opt_acc = accuracy_score(y_pred=preds_opt, y_true=labels_opt, normalize=True, sample_weight=None)

    count_equal_ans = sum(x == y for x, y in zip(preds_num, labels_num))
    num_acc = round(count_equal_ans/len(labels_num)*100, 4)

    micro_f1 = round(micro_f1['f1']*100, 4)
    macro_f1 = round(macro_f1['f1']*100, 4)

    print(f"micro_f1: {micro_f1}")
    print(f"macro_f1: {macro_f1}")
    print(f"opt_acc: {opt_acc}")
    print(f"num_acc: {num_acc}")

    save_res = [{"news_article": sample["news_article"], "question_stem": sample["question_stem"], "ans": sample['ans'], "target_num": sample['target_num']} for sample in dataset_test]

    for res, pred_opt, pred_num, pred_out in zip(save_res, preds_opt, preds_num, preds_out):
        res['pred_opt'] =  pred_opt
        res['pred_num'] =  pred_num
        res['pred_out'] = pred_out

    json_file_path = os.path.join(args.output_dir, args.output_file_name)

    print("save predict res to: "+json_file_path)
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(save_res, json_file, ensure_ascii=False)


def run(args):
    def preprocess_function(sample):
        inputs = [input_template.format(news_article=news_article.strip(), question=question.strip(), option1=str(answer_options[0]), option2=str(answer_options[1]), option3=str(answer_options[2]), option4=str(answer_options[3])) for news_article, question, answer_options  in zip(sample["news_article"], sample['question_stem'], sample['answer_options'])]

        model_inputs = tokenizer(inputs, truncation=False)

        labels = [label_template.format(ans=covert_dic[ans], num=num) for ans, num in zip(sample["ans"], sample['target_num'])]

        model_labels = tokenizer(text_target=labels, truncation=False)

        model_inputs["labels"] = model_labels["input_ids"]
        return model_inputs

    covert_dic = {'0':'A', '1':'B', '2':'C', '3':'D'}

    set_seed(args.seed)

    nsqad_template = instr_template()
    nsqad_template.load_nsqad_template()

    input_template = nsqad_template.input_template['icl_ch']

    label_template = nsqad_template.label_template['ans']

    model_name = args.model_name
    data_train_pth = args.data_train_pth
    data_test_pth = args.data_test_pth
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    datasets = DatasetDict()

    if args.task == "train":
        dataset_train = read_jsonl(data_train_pth)[0]
        data_split = int(len(dataset_train)*0.9)
        random.seed(args.seed)
        random.shuffle(dataset_train)

        dataset_dev = dataset_train[data_split:]
        dataset_train = dataset_train[:data_split]
        
        dataset_train = Dataset.from_dict(transToDict(dataset_train))
        dataset_dev = Dataset.from_dict(transToDict(dataset_dev))

        datasets['train'] = dataset_train
        datasets['dev'] = dataset_dev
        
        tokenized_dataset = datasets.map(preprocess_function, batched=True, remove_columns=['news_article', 'question_stem', 'answer_options', 'ans', 'target_num'])
        train_and_evaluate(args, tokenizer, tokenized_dataset)
    else:
        dataset_test = read_jsonl(data_test_pth)[0]
        dataset_test = Dataset.from_dict(transToDict(dataset_test))
        datasets['test'] = dataset_test
        tokenized_dataset = datasets.map(preprocess_function, batched=True, remove_columns=['news_article', 'question_stem', 'answer_options', 'ans', 'target_num'])
        predict_and_save_res(args, tokenizer, tokenized_dataset, dataset_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training code")
    parser.add_argument("--data_train_pth", default='./NQuAD/NQuAD_train.json', help="dataset_train's path")
    parser.add_argument("--data_test_pth", default='./NQuAD/NQuAD_test.json', help="dataset_test's path")
    parser.add_argument("--has_dev", default=True, help="whether has dev dataset")
    parser.add_argument("--model_name", default='google/mt5-small', help="model name")
    parser.add_argument("--seed", default=42, help="set seed")
    parser.add_argument("--model_checkpoint", default='', help="model checkpoint's path")
    parser.add_argument("--task", default='train', help="train or predict")
    parser.add_argument("--evaluation_strategy", default='epoch', help="evaluation_strategy")
    parser.add_argument("--save_strategy", default='epoch', help="save_strategy")
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warm_up_radio', type=float, default=0.1)
    parser.add_argument('--gradient_accumulation_steps', default=10, help='gradient_accumulation')
    parser.add_argument('--num_train_epochs', default=10)
    parser.add_argument('--output_model_path', type=str, default='./nsqad-model')
    parser.add_argument('--weight_decay', default=0.01, help='dropout_rate')
    parser.add_argument("--output_file_name", default="save_res_qnli.json", help="output file's name")
    parser.add_argument("--output_dir", default="save_res", help="output file's dir")
    args = parser.parse_args()
     
    run(args)
