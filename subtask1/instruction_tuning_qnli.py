from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from sklearn.model_selection import KFold

import json

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import random, json, re
from random import randrange, sample
import argparse
import os

import evaluate
import numpy as np
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from util import *

from instruction_config import *

def train_and_evaluate(args, tokenizer, tokenized_dataset):
    def compute_metrics(eval_pred):
        covert_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [covert_dict.get(item) for item in decoded_preds]
        decoded_labels = [covert_dict.get(item) for item in decoded_labels]

        macro_f1 = f1_metric.compute(predictions=decoded_preds, references=decoded_labels, average="macro")
        micro_f1 = f1_metric.compute(predictions=decoded_preds, references=decoded_labels, average="micro")

        result = {}
        result['macro_f1'] = macro_f1['f1']*100
        result['micro_f1'] = micro_f1['f1']*100

        if best_micro_f1['f1'] < micro_f1['f1']*100:
            best_micro_f1['f1'] = micro_f1['f1']*100

        return result
    
    best_micro_f1 = {}
    best_micro_f1['f1'] = 0.0

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
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics
    )

    torch.cuda.empty_cache()
    trainer.train()

    print(f"Trainging end..")
    print(f"Best_f1 for split-{args.select_split_idx} is {best_micro_f1['f1']}")

def run(args):
    def preprocess_function(sample):
        if args.is_digit_base:
            inputs = [input_template.format(statement1=statement1.strip(), statement2=statement2.strip(), options=options.lower().strip()) for statement1, statement2, options in zip(sample["statement1_char"], sample["statement2_char"], sample['options'])]
        else:
            inputs = [input_template.format(statement1=statement1.strip(), statement2=statement2.strip(), options=options.lower().strip()) for statement1, statement2, options in zip(sample["statement1"], sample["statement2"], sample['options'])]

        model_inputs = tokenizer(inputs, truncation=False)

        labels = [answer.strip().lower() for answer in sample["answer"]]

        model_labels = tokenizer(text_target=labels, truncation=False)

        model_inputs["labels"] = model_labels["input_ids"]
        return model_inputs

    model_name = args.model_name
    data_train_pth = args.data_train_pth

    set_seed(args.seed)

    qnli_template = instr_template()
    qnli_template.load_qnli_template()

    if args.has_demonstrations == True:
        input_template = qnli_template.input_template['icl']
    else:
        input_template = qnli_template.input_template['instr']

    data = read_jsonl(data_train_pth)[0]

    kf = KFold(n_splits=args.num_splits, shuffle=False)

    select_idx = 0
    for train_index, test_index in kf.split(data):
        if select_idx >= args.select_split_idx:
            break
        dataset_train = [data[i] for i in train_index]
        dataset_test = [data[i] for i in test_index]
        select_idx = select_idx + 1

    datasets = DatasetDict()
    dataset_train = Dataset.from_dict(trans_to_dict_qnli(dataset_train))
    dataset_test = Dataset.from_dict(trans_to_dict_qnli(dataset_test))

    datasets['train'] = dataset_train
    datasets['test'] = dataset_test

    tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    tokenized_dataset = datasets.map(preprocess_function, batched=True, remove_columns=["statement1", "statement2","statement1_char", "statement2_char", "options", "answer"])
    train_and_evaluate(args, tokenizer, tokenized_dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training code")
    parser.add_argument("--data_train_pth", default='./Quantitative-101/QNLI/NewsNLI.json', help="dataset_train's path")
    parser.add_argument("--num_splits", default=10, help="num of splits")
    parser.add_argument("--select_split_idx", default=2, help="select which split to evaluate")
    parser.add_argument("--is_digit_base", default=False, help="whether to use digit")
    parser.add_argument("--has_demonstrations", default=True, help="whether has demonstrations")
    parser.add_argument("--model_name", default='google/flan-t5-base', help="model name")
    parser.add_argument("--seed", default=42, help="set seed")
    parser.add_argument("--model_checkpoint", default='', help="model checkpoint's path")
    parser.add_argument("--task", default='eval', help="train or predict")
    parser.add_argument("--evaluation_strategy", default='epoch', help="evaluation_strategy")
    parser.add_argument("--save_strategy", default='no', help="save_strategy")
    parser.add_argument('--per_device_train_batch_size', type=int, default=10)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-7)
    parser.add_argument('--warm_up_radio', type=float, default=0.1)
    parser.add_argument('--gradient_accumulation_steps', default=1, help='gradient_accumulation')
    parser.add_argument('--num_train_epochs', default=30)
    parser.add_argument('--output_model_path', type=str, default='./qnli_model')
    parser.add_argument('--weight_decay', default=0.01, help='dropout_rate')
    parser.add_argument("--output_file_name", default="qnli_res.json", help="output file's name")
    parser.add_argument("--output_dir", default="save_res", help="output file's dir")
    args = parser.parse_args()
     
    run(args)