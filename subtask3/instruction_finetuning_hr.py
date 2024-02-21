from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, pipeline
from datasets import Dataset, DatasetDict

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import random, json, re
from random import randrange
import argparse
import os

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType, PeftModel, PeftConfig

import evaluate
import numpy as np
from tqdm import tqdm

from util import *

from instruction_config import *

def train_and_evaluate(args, tokenizer, tokenized_dataset):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        if predict_type['is_cot']:
            decode_labels_ans = batch_find_ans(decoded_labels)
            decode_pred_ans = batch_find_ans(decoded_preds)
        else:
            decode_labels_ans = decoded_labels
            decode_pred_ans = decoded_preds

        count_equal_ans = sum(x == y for x, y in zip(decode_labels_ans, decode_pred_ans))

        num_acc = round(count_equal_ans/len(decode_labels_ans)*100, 4)
        result = {}
        result['num_acc'] = num_acc
        return result

    predict_type = {}
    predict_type['is_cot'] = args.is_cot
    large_scale = False
    call_back = []
    if args.model_name in ['google/flan-t5-xxl', 'google/flan-t5-xl']:
        large_scale = True
        call_back.append(PeftSavingCallback)

    if large_scale:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, load_in_8bit=True, use_cache=False, device_map="auto")
        lora_config = LoraConfig(
            r=8, 
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

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
        fp16=large_scale,
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
        compute_metrics=compute_metrics,
        callbacks=call_back
    )

    torch.cuda.empty_cache()
    # Start training
    trainer.train()

def predict_and_save_res(args, tokenizer=None, tokenized_dataset=None, dataset_test=None):
    def get_predict(model, tokenized_dataset, batch_size = 4, max_length = 128, sample_set = 'test', device = 'cuda'):
        """
        Get the predictions from the trained model.
        """
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            return input_ids
        
        dataloader = DataLoader(tokenized_dataset[sample_set], batch_size=batch_size, collate_fn=collate_fn)
        predicted_output = []
        pred_ans = []
        model.to(device)
        print('Model loaded to: ', device)

        for inputs in tqdm(dataloader):
            inputs = inputs.to(device)
            output_ids = model.generate(input_ids=inputs, max_new_tokens=max_length)
            output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            if args.is_cot:
                decode_pred_ans = batch_find_ans(output_texts)
            else:
                decode_pred_ans = output_texts

            for ans, pred in zip(decode_pred_ans, output_texts):
                predicted_output.append(pred)
                pred_ans.append(ans)
            
        return predicted_output, pred_ans

    if args.model_name in ['google/flan-t5-xxl', 'google/flan-t5-xl']:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
    else:
        config = PeftConfig.from_pretrained(args.model_checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path,  load_in_8bit=True,  device_map={'': 0})
        model = PeftModel.from_pretrained(model, args.model_checkpoint, device_map={'': 0})

    model.eval()

    predicted_output, pred_ans = get_predict(model=model, tokenized_dataset = tokenized_dataset, batch_size=args.per_device_eval_batch_size, max_length=512, sample_set='test')

    label_ans = [sample["ans"] for sample in dataset_test]
    label_types = [1 if "Paraphrase" in sample['calculation'] or "Round" in sample["calculation"] or "Subtract" in sample['calculation'] or "Add" in sample['calculation'] or "Span" in sample['calculation'] or "Divide" in sample['calculation'] or "Multiply" in sample['calculation'] else 0 for sample in dataset_test]

    save_res = [{"news": sample["news"], "masked headline": sample["masked headline"], "calculation": sample["calculation"], "ans": sample["ans"]} for sample in dataset_test]

    label_ans_copy = []
    label_ans_reason = []

    pred_ans_copy = []
    pred_ans_reason = []

    for ans, pred, tp in zip(label_ans,pred_ans,label_types):
        if tp == 0:
            label_ans_copy.append(ans)
            pred_ans_copy.append(pred)
        else:
            label_ans_reason.append(ans)
            pred_ans_reason.append(pred)

    count_equal_ans = sum(x == y for x, y in zip(label_ans, pred_ans))
    count_equal_ans_copy = sum(x == y for x, y in zip(label_ans_copy, pred_ans_copy))
    count_equal_ans_reason = sum(x == y for x, y in zip(label_ans_reason, pred_ans_reason))

    num_acc = round(count_equal_ans/len(label_ans)*100, 3)
    num_acc_copy = round(count_equal_ans_copy/len(label_ans_copy)*100, 3)
    num_acc_reason = round(count_equal_ans_reason/len(label_ans_reason)*100, 3)

    print(f"Num_acc: {num_acc}") 
    print(f"Num_acc_copy: {num_acc_copy}")
    print(f"Num_acc_reason: {num_acc_reason}")

    for res, ans, pred in zip(save_res, pred_ans, predicted_output):
        res['pred_ans'] = ans
        res['pred_cot'] = pred

    os.makedirs(args.output_dir, exist_ok=True)

    json_file_path = os.path.join(args.output_dir, args.output_file_name)

    print("save predict res to: "+json_file_path)
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(save_res, json_file, ensure_ascii=False)

def run(args):
    def preprocess_function(sample):
        # add prefix to the input
        inputs = [input_template.format(news = news, mskh = mskh) for news, mskh in zip(sample["news"], sample["masked headline"])]

        model_inputs = tokenizer(inputs, truncation=False)

        if not args.is_cot or args.task != "train": # test set don't contain generate_template
            labels = [label_template.format(calculation=cal, ans=ans) for cal, ans in zip(sample["calculation"], sample["ans"])]
        else:
            labels = [label_template.format(cot = cot, ans = ans) for cot, ans in zip(sample["generate_template"], sample["ans"])]

        model_labels = tokenizer(text_target=labels, truncation=False)

        model_inputs["labels"] = model_labels["input_ids"]
        
        return model_inputs

    set_seed(args.seed)

    hr_template = instr_template()
    hr_template.load_hr_template()

    if args.is_cot:
        input_template = hr_template.input_template['cot']
        label_template = hr_template.label_template['cot']
    else:
        input_template = hr_template.input_template['opt']
        label_template = hr_template.label_template['opt']

    model_name = args.model_name
    data_train_pth = args.data_train_pth
    data_dev_pth = args.data_dev_pth
    data_test_pth = args.data_test_pth
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    datasets = DatasetDict()

    if args.task == "train":
        dataset_train = read_jsonl(data_train_pth)[0]
        dataset_train = Dataset.from_dict(dropout_redundant_hr(dataset_train))

        datasets['train'] = dataset_train
        if args.has_dev:
            dataset_dev = read_jsonl(data_dev_pth)[0]
            dataset_dev = Dataset.from_dict(dropout_redundant_hr(dataset_dev))
            datasets['dev'] = dataset_dev
        else:
            dataset_test = read_jsonl(data_test_pth)[0]
            dataset_test = Dataset.from_dict(dropout_redundant_hr(dataset_test))
            datasets['dev'] = dataset_test
        
        tokenized_dataset = datasets.map(preprocess_function, batched=True, remove_columns=["news", "headline", "similar_news", "similar_headline"])
        train_and_evaluate(args, tokenizer, tokenized_dataset)
    else:
        dataset_test = read_jsonl(data_test_pth)[0]
        dataset_test = Dataset.from_dict(dropout_redundant_hg(dataset_test))
        datasets['test'] = dataset_test
        tokenized_dataset = datasets.map(preprocess_function, batched=True, remove_columns=["news", "headline", "similar_news", "similar_headline"])
        predict_and_save_res(args, tokenizer, tokenized_dataset, dataset_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training code")
    parser.add_argument("--data_train_pth", default='./numeval_datasets/Train_Numerical_Reasoning_Template_CoT.json', help="dataset_train's path")
    parser.add_argument("--data_dev_pth", default='./numeval_datasets/Dev_Numerical_Reasoning_Template_CoT.json', help="dataset_dev's path")
    parser.add_argument("--data_test_pth", default='./numeval_datasets/ANS-Test_Numerical_Reasoning.json', help="dataset_test's path")
    parser.add_argument("--has_dev", default=True, help="whether has dev dataset")
    parser.add_argument("--is_cot", default=False, help="whether has demonstrations")
    parser.add_argument("--model_name", default='google/flan-t5-base', help="model name")
    parser.add_argument("--seed", default=42, help="set seed")
    parser.add_argument("--model_checkpoint", default='', help="model checkpoint's path")
    parser.add_argument("--task", default='train', help="train or predict")
    parser.add_argument("--evaluation_strategy", default='epoch', help="evaluation_strategy")
    parser.add_argument("--save_strategy", default='epoch', help="save_strategy")
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warm_up_radio', type=float, default=0.1)
    parser.add_argument('--gradient_accumulation_steps', default=1, help='gradient_accumulation')
    parser.add_argument('--num_train_epochs', default=10)
    parser.add_argument('--output_model_path', type=str, default='/root/autodl-tmp/hr_model')
    parser.add_argument('--weight_decay', default=0.01, help='dropout_rate')
    parser.add_argument("--output_file_name", default="hg_res.json", help="output file's name")
    parser.add_argument("--output_dir", default="save_res", help="output file's dir")
    args = parser.parse_args()
     
    run(args)
