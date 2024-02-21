from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

import json

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import argparse
import os

import evaluate
import numpy as np
from tqdm import tqdm

from util import *

from instruction_config import *

def train_and_evaluate(args, tokenizer, tokenized_dataset):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        covert_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
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
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            attention_mask = [torch.tensor(example['attention_mask']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=tokenizer.pad_token_id)
            
            return input_ids, attention_mask
        
        dataloader = DataLoader(tokenized_dataset[sample_set], batch_size=batch_size, collate_fn=collate_fn)
        model.to(device)
        print('Model loaded to: ', device)
        preds = []

        for inputs, attention_mask in tqdm(dataloader):
            inputs = inputs.to(device)
            attention_mask = attention_mask.to(device)

            output_ids = model.generate(input_ids=inputs, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
            
            decode_pred_ans = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            preds += decode_pred_ans
        
        preds = [covert_dict.get(item) for item in preds]
        
        return preds

    covert_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    f1_metric = evaluate.load("./f1.py")

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)

    decoded_preds = get_predict(model=model, tokenized_dataset = tokenized_dataset, batch_size=25, max_new_tokens=25, sample_set='test', device = 'cuda')

    labels = [covert_dict.get(sample["answer"].lower().strip()) for sample in dataset_test]

    macro_f1 = f1_metric.compute(predictions=decoded_preds, references=labels, average="macro")
    micro_f1 = f1_metric.compute(predictions=decoded_preds, references=labels, average="micro")

    micro_f1 = round(micro_f1['f1']*100, 4)
    macro_f1 = round(macro_f1['f1']*100, 4)

    print(f"micro_f1: {micro_f1}")
    print(f"macro_f1: {macro_f1}")

    save_res = [{"statement1": sample["statement1"], "statement2": sample["statement2"], "options": sample['options'], "answer": sample['answer']} for sample in dataset_test]

    for res, preds in zip(save_res, decoded_preds):
        res['preds'] =  preds

    json_file_path = os.path.join(args.output_dir, args.output_file_name)

    print("save predict res to: "+json_file_path)
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(save_res, json_file, ensure_ascii=False)

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

    set_seed(args.seed)

    qnli_template = instr_template()
    qnli_template.load_qnli_template()

    if args.has_demonstrations == True:
        input_template = qnli_template.input_template['icl']
    else:
        input_template = qnli_template.input_template['instr']

    model_name = args.model_name
    data_train_pth = args.data_train_pth
    data_dev_pth = args.data_dev_pth
    data_test_pth = args.data_test_pth
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    datasets = DatasetDict()

    if args.task == "train":
        dataset_train = read_jsonl(data_train_pth)[0]
        dataset_train = Dataset.from_dict(trans_to_dict_qnli(dataset_train))

        datasets['train'] = dataset_train
        if args.has_dev:
            dataset_dev = read_jsonl(data_dev_pth)[0]
            dataset_dev = Dataset.from_dict(trans_to_dict_qnli(dataset_dev))
            datasets['dev'] = dataset_dev
        else:
            dataset_test = read_jsonl(data_test_pth)[0]
            dataset_test = Dataset.from_dict(trans_to_dict_qnli(dataset_test))
            datasets['dev'] = dataset_test
        
        tokenized_dataset = datasets.map(preprocess_function, batched=True, remove_columns=["statement1", "statement2","statement1_char", "statement2_char", "options", "answer"])
        train_and_evaluate(args, tokenizer, tokenized_dataset)
    else:
        dataset_test = read_jsonl(data_test_pth)[0]
        dataset_test = Dataset.from_dict(trans_to_dict_qnli(dataset_test))
        datasets['test'] = dataset_test
        tokenized_dataset = datasets.map(preprocess_function, batched=True, remove_columns=["statement1", "statement2","statement1_char", "statement2_char", "options", "answer"])
        predict_and_save_res(args, tokenizer, tokenized_dataset, dataset_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training code")
    parser.add_argument("--data_train_pth", default='./Quantitative-101/QNLI/QNLI-Stress Test/QNLI-Stress Test_train.json', help="dataset_train's path")
    parser.add_argument("--data_dev_pth", default='./Quantitative-101/QNLI/QNLI-Stress Test/QNLI-Stress Test_dev.json', help="dataset_dev's path")
    parser.add_argument("--data_test_pth", default='./Quantitative-101/QNLI/QNLI-Stress Test/QNLI-Stress Test_test.json', help="dataset_test's path")
    parser.add_argument("--is_digit_base", default=False, help="whether to use digit")
    parser.add_argument("--has_dev", default=True, help="whether has dev dataset")
    parser.add_argument("--has_demonstrations", default=False, help="whether has demonstrations")
    parser.add_argument("--model_name", default='google/flan-t5-base', help="model name")
    parser.add_argument("--seed", default=42, help="set seed")
    parser.add_argument("--model_checkpoint", default='', help="model checkpoint's path")
    parser.add_argument("--task", default='eval', help="train or predict")
    parser.add_argument("--evaluation_strategy", default='epoch', help="evaluation_strategy")
    parser.add_argument("--save_strategy", default='epoch', help="save_strategy")
    parser.add_argument('--per_device_train_batch_size', type=int, default=20)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--warm_up_radio', type=float, default=0.1)
    parser.add_argument('--gradient_accumulation_steps', default=1, help='gradient_accumulation')
    parser.add_argument('--num_train_epochs', default=20)
    parser.add_argument('--output_model_path', type=str, default='./qnli_model')
    parser.add_argument('--weight_decay', default=0.01, help='dropout_rate')
    parser.add_argument("--output_file_name", default="save_res_qnli.json", help="output file's name")
    parser.add_argument("--output_dir", default="save_res", help="output file's dir")
    args = parser.parse_args()
     
    run(args)