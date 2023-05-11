import pickle5 as pickle
import argparse
import json
import numpy as np
import pandas as pd
import os
import ray
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, default_data_collator
from datasets import load_dataset, load_metric
import evaluate
import ray.data
from ray.data.preprocessors import BatchMapper
from ray.train.huggingface import HuggingFaceTrainer
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.air.integrations.wandb import WandbLoggerCallback
import wandb


os.environ['RAY_memory_usage_threshold'] = "0.98"

def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')

    parser.add_argument('--dataset', type=str, default="imdb",
                        help='Dataset to finetune model on')
    parser.add_argument('--model', type=str, default="distillbert",
                        help = "Model to finetune")
    parser.add_argument('--num_labels', type=int, default=2,
                        help = "Number of classes to predict")

    parser.add_argument('--gpu', default=True,
                        action="store_true",
                        help='gpu used or not')
    parser.add_argument('--num_workers', default = 2, type=int,
                        help = "Number of gpus to use")

    parser.add_argument('--cpus_per_worker', default = 1, type=int,
                        help = "Number of CPUs per GPU")

    parser.add_argument('--batch_size', default = 8, type=int,
                        help = "Batch size for finetuning")

    parser.add_argument('--deepspeed_config', default = "config/deepspeed.json", 
                        help = "Deepspeed config file")

    parser.add_argument('--wandb_project_name', default = "bert-finetuning",
                        help = "wandb project name")

    args = parser.parse_args()

    return args

def initialize_ray():

    ray.init(
    runtime_env={
        "pip": [
            "datasets",
            "evaluate",
            "accelerate>=0.16.0",
            "transformers>=4.26.0",
            "torch>=1.12.0",
            "deepspeed",
        ]
    }
)

def load_ray_dataset(dataset_name):
    huggingface_dataset = load_dataset(dataset_name)
    ray_datasets = ray.data.from_huggingface(huggingface_dataset)
    return ray_datasets

def tokenize_bert(batch):
    """
    Tokenizes data according to Bert specifications and returns the 
    tokenized ids and attention masks.
    """
    # Tokenize the input text
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenized_texts = tokenizer(list(batch['text']), padding=True, truncation=True, return_tensors="np")
    tokenized_texts['input_ids'] = tokenized_texts['input_ids'].tolist()
    tokenized_texts['attention_mask'] = tokenized_texts['attention_mask'].tolist()
    tokenized_texts = {**batch, **tokenized_texts}
    return pd.DataFrame.from_dict(tokenized_texts)


def batch_process_bert_data():
        return BatchMapper(tokenize_bert, batch_format = "pandas")


def trainer_init_per_worker(train_dataset, eval_dataset = None, **config):
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    torch.backends.cuda.matmul.allow_tf32 = True
    deepspeed = deepspeed_config
    args = TrainingArguments(
        "ray-bert-finetune-imdb",
        save_strategy="no",
        logging_strategy="epoch",
        # logging_steps=5,
        evaluation_strategy = "epoch",
        learning_rate=config.get("learning_rate", 2e-5),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
#         label_names=["input_ids", "attention_mask"],
        num_train_epochs=config.get("epochs", 2),
        weight_decay=config.get("weight_decay", 0.01),
        fp16=True,
        # gradient_checkpointing=True,
        deepspeed=deepspeed,
        push_to_hub=False,
        disable_tqdm=True,  # declutter the output a little
        # no_cuda=not use_gpu,  # you need to explicitly set no_cuda if you want CPUs
    )
    def compute_metrics(eval_pred):
        load_accuracy = evaluate.load("accuracy")
        load_f1 = evaluate.load("f1")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": accuracy, "f1": f1}


    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator = default_data_collator,
    )

    print("Starting training")
    return trainer

def get_ray_trainer(ray_dataset, num_workers, wandb_project_name, use_gpu=True):
    trainer = HuggingFaceTrainer(
    trainer_init_per_worker=trainer_init_per_worker,
    scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu, resources_per_worker={"GPU": 1, "CPU": 1}),
    datasets={
        "train": ray_dataset["train"],
        "evaluation": ray_dataset["test"],
    },
    run_config=RunConfig(
        callbacks=[WandbLoggerCallback(project=wandb_project_name)],
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min",
        ),
        # local_dir="/tmp/ars2352/"
    ),
    preprocessor=batch_process_bert_data(),
)
    return trainer



def finetune(ray_dataset, args):

    trainer = get_ray_trainer(ray_dataset,
                              args.num_workers,
                              args.wandb_project_name,
                              args.gpu)

    result = trainer.fit()


if __name__ == '__main__':
    args = parse_option()
    deepspeed_config = {
    "fp16": {
        "enabled": "auto",
        "initial_scale_power": 8
    },
    "bf16": {"enabled": "auto"},
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "gather_16bit_weights_on_model_save": True,
        "round_robin_gradients": True
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 10,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}
    model_name = args.model
    num_labels = args.num_labels
    batch_size = args.batch_size
    ### Initialize ray environment and load dependencies for each node in Ray
    initialize_ray()

    ### Create Ray dataset for parallel processing of data    
    ray_dataset = load_ray_dataset(args.dataset)

    finetune(ray_dataset, args)











