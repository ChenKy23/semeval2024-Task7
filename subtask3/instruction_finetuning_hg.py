from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, pipeline
from datasets import Dataset, DatasetDict

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import random, json, re
from random import randrange
import argparse
import os

from moverscore_v2 import word_mover_score, get_idf_dict

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType, PeftModel, PeftConfig

from bert_score import score

import evaluate
import numpy as np
from tqdm import tqdm

from util import *

from instruction_config import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    large_scale = False
    call_back = None
    if args.model_name in ['google/flan-t5-xxl', 'google/flan-t5-xl']:
        large_scale = True
        call_back= [PeftSavingCallback]

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

    metric = evaluate.load("./rouge.py")

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
    def get_predict_and_labels(model, tokenized_dataset, batch_size = 4, max_length = 128, sample_set = 'test'):
        """
        Get the predictions from the trained model.
        """
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            return input_ids
        
        dataloader = DataLoader(tokenized_dataset[sample_set], batch_size=batch_size, collate_fn=collate_fn)
        predicted_output = []

        for inputs in tqdm(dataloader):
            inputs = inputs.to('cuda')
            output_ids = model.generate(inputs, max_length=max_length, num_beams = 5)
            output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output_text in output_texts:
                predicted_output.append(output_text)
            
        return predicted_output

    metric = evaluate.load("./rouge.py")

    if args.model_name in ['google/flan-t5-xxl', 'google/flan-t5-xl']:
        config = PeftConfig.from_pretrained(args.model_checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path,  load_in_8bit=True,  device_map={'': 0})
        model = PeftModel.from_pretrained(model, args.model_checkpoint, device_map={'': 0})
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint)
        model.to('cuda')

    predicted_output = get_predict_and_labels(model=model, tokenized_dataset = tokenized_dataset, batch_size=args.per_device_eval_batch_size, max_length=64, sample_set='test')

    labels_output = [sample['headline'] for sample in dataset_test]

    rogue = metric.compute(predictions=predicted_output, references=labels_output, use_stemmer=True)

    # print results 
    print(f"Rogue1: {rogue['rouge1']* 100:2f}%")
    print(f"rouge2: {rogue['rouge2']* 100:2f}%")
    print(f"rougeL: {rogue['rougeL']* 100:2f}%")
    print(f"rougeLsum: {rogue['rougeLsum']* 100:2f}%")

    P, R, F1 = score(predicted_output, labels_output, batch_size=4, device='cuda', lang='en', rescale_with_baseline=True, use_fast_tokenizer=True)

    print(f"System level P score: {P.mean()*100:.3f}")
    print(f"System level R score: {R.mean()*100:.3f}")
    print(f"System level F1 score: {F1.mean()*100:.3f}")

    idf_dict_hyp = get_idf_dict(predicted_output)
    idf_dict_ref = get_idf_dict(labels_output)
    mover_score = word_mover_score(labels_output, predicted_output, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)
    mover = np.mean(mover_score)
    print("MoverScore: %.6f"%(mover))

    cal_num_acc(predicted_output, args.num_gt_path, args.num_type_path)

    save_res = [{"news": sample["news"], "headline": sample["headline"]} for sample in dataset_test]

    for res, headline in zip(save_res, predicted_output):
        res['generation'] = headline

    os.makedirs(args.output_dir, exist_ok=True)

    json_file_path = os.path.join(args.output_dir, args.output_file_name)

    print("save predict res to: "+json_file_path)
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(save_res, json_file, ensure_ascii=False)


def run(args):
    def preprocess_function(sample):
        # add prefix to the input
        if args.has_demonstrations:
            inputs = [input_template.format(similar_news=similar_news, similar_headline=similar_headline, news=news) for similar_news, similar_headline, news in zip(sample["similar_news"], sample["similar_headline"], sample["news"])]
        else:
            inputs = [input_template.format(news=news) for news in sample["news"]]
        # tokenize inputs
        model_inputs = tokenizer(inputs, truncation=False)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=sample["headline"], truncation=False)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    set_seed(args.seed)

    hg_template = instr_template()
    hg_template.load_hg_template()

    if args.has_demonstrations:
        input_template = hg_template.input_template['icl']
    else:
        input_template = hg_template.input_template['instr']

    model_name = args.model_name
    data_train_pth = args.data_train_pth
    data_dev_pth = args.data_dev_pth
    data_test_pth = args.data_test_pth
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    datasets = DatasetDict()

    if args.task == "train":
        dataset_train = read_jsonl(data_train_pth)[0]
        dataset_train = Dataset.from_dict(dropout_redundant_hg(dataset_train))

        datasets['train'] = dataset_train
        if args.has_dev:
            dataset_dev = read_jsonl(data_dev_pth)[0]
            dataset_dev = Dataset.from_dict(dropout_redundant_hg(dataset_dev))
            datasets['dev'] = dataset_dev
        else:
            dataset_test = read_jsonl(data_test_pth)[0]
            dataset_test = Dataset.from_dict(dropout_redundant_hg(dataset_test))
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
    parser.add_argument("--data_train_pth", default='./numeval_datasets/Train_Headline_Generation_Similarity_Search.json', help="dataset_train's path")
    parser.add_argument("--data_dev_pth", default='./numeval_datasets/Dev_Headline_Generation_Similarity_Search.json', help="dataset_dev's path")
    parser.add_argument("--data_test_pth", default='./numeval_datasets/ANS-Test_Headline_Generation_Similarity_Search.json', help="dataset_test's path")
    parser.add_argument("--num_gt_path", default="./numeval_datasets/number_gt.txt", type=str, help="numerical ground truth path")
    parser.add_argument("--num_type_path", default="./numeval_datasets/number_type.txt", type=str, help="type of each summary, 1:Reasoning, 0:Copy")
    parser.add_argument("--has_dev", default=True, help="whether has dev dataset")
    parser.add_argument("--has_demonstrations", default=False, help="whether has demonstrations")
    parser.add_argument("--model_name", default='google/flan-t5-base', help="model name")
    parser.add_argument("--seed", default=42, help="set seed")
    parser.add_argument("--model_checkpoint", default='', help="model checkpoint's path")
    parser.add_argument("--task", default='eval', help="train or predict")
    parser.add_argument("--evaluation_strategy", default='epoch', help="evaluation_strategy")
    parser.add_argument("--save_strategy", default='epoch', help="save_strategy")
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warm_up_radio', type=float, default=0.1)
    parser.add_argument('--gradient_accumulation_steps', default=1, help='gradient_accumulation')
    parser.add_argument('--num_train_epochs', default=10)
    parser.add_argument('--output_model_path', type=str, default='./hg_model')
    parser.add_argument('--weight_decay', default=0.01, help='dropout_rate')
    parser.add_argument("--output_file_name", default="hg_res.json", help="output file's name")
    parser.add_argument("--output_dir", default="save_res", help="output file's dir")
    args = parser.parse_args()
     
    run(args)
