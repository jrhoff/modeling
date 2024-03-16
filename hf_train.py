import yaml
import os
import wandb
from typing import Dict, Any
import torch
from pathlib import Path
from argparse import ArgumentParser
import transformers
from transformers import Trainer, AutoTokenizer, TrainingArguments
from modeling.peft_training import PeftHelper
from modeling import collators

# TODO: implement w. jsonargparse & custom parsers for descript training arguments later
# from jsonargparse import CLI

os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

def load_model_and_tokenizer(model_name_or_path:str, base_class: str):
    model_class = getattr(transformers, base_class)
    model = model_class.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def modeling_sanity_checks():
    return True



def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    # linear warmup followed by cosine annealing
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(max_steps - warmup_steps))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[warmup_steps])

def get_optimizer(model, learning_rate, weight_decay):
    # TODO: can make this configable, but not top priority
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(learning_rate), weight_decay=float(weight_decay))
    return optimizer

def get_callbacks(config):
    callbacks = []
    for callback in config.get("callbacks", []):
        callback.append(
            getattr(transformers, callback["name"])(**callback["arguments"])
            )
    return callbacks

def get_collator(config, tokenizer):
    collator_arguments = config["collator_arguments"]
    collator_class = getattr(collators, collator_arguments["class_name"])
    return collator_class(tokenizer, **collator_arguments.get("class_arguments", {}))


def parse_config(config):
    # TODO: once we have parsers for diff classes
    return config

def set_environment_variables(config):
    if variables := config.get("environment_variables", False):
        for k, v in variables.items():
            os.environ[k] = v


def prepare_model_artifacts(config: Dict[str, Any]):
    model_arguments = config["model_arguments"]

    # load base model
    model_name_or_path = model_arguments["model_name_or_path"]
    base_class = model_arguments["base_class"]

    model, tokenizer = load_model_and_tokenizer(model_name_or_path, base_class)

    # need to handle peft?
    if peft_arguments := config.get("peft_arguments", False):
        peft_helper = PeftHelper(peft_arguments)
        model = peft_helper.get_model(model)
    return model, tokenizer


def get_datasets(config: Dict[str, Any]):
    data_arguments = config["data_arguments"]

    # data is expected to be in .pt files
    train = None
    if train_loc := data_arguments.get("train_location", False):
        with open(train_loc, "rb") as f:
            train = torch.load(f)

    eval = None
    if eval_loc := data_arguments.get("eval_location", False):
        with open(train_loc, "rb") as f:
            eval = torch.load(f)

    return {
        "train": train,
        "eval": eval[:10]
    }

def calculate_warmup_steps(datasets: int, warmup_ratio: int):
    train_size = datasets["train"].__len__()
    return round(train_size * warmup_ratio)


def main(config: Path):
    with open(config, "r") as reader:
        config_location = config
        config = yaml.full_load(reader)
    
    modeling_sanity_checks()  # validate args

    model, tokenizer = prepare_model_artifacts(config)
    training_arguments = TrainingArguments(**config["training_arguments"])


    datasets = get_datasets(config)
    collator = get_collator(config, tokenizer)
    optimizer = get_optimizer(model, training_arguments.learning_rate, training_arguments.weight_decay)
    
    scheduler = get_lr_scheduler(optimizer, 
                                 warmup_steps=calculate_warmup_steps(datasets, training_arguments.warmup_ratio),
                                 max_steps=training_arguments.max_steps)
    
    callbacks = get_callbacks(config)

    
    run = None
    if wandb_arguments := config.get("wandb_arguments", False):
        project = wandb_arguments["project"]
        os.environ["WANDB_PROJECT"] = project
        wandb.login(key = wandb_arguments["key"], relogin=False)
        run = wandb.init(
            project=project, 
            job_type="training"
        )
        run.config.update(config)

        artifact = wandb.Artifact(name="config_file", type="file")
        artifact.add_file(local_path=config_location)  # Add dataset directory to artifact
        run.log_artifact(artifact)  # Logs the artifact version "my_data:v0


    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        args=training_arguments,
        data_collator=collator,
        optimizers=(optimizer, scheduler),
        callbacks=callbacks
    )

    

    trainer.train()

    wandb.finish()

if __name__ == "__main__":
    # CLI(main)
    parser = ArgumentParser()
    parser.add_argument("config", type=Path)
    main(parser.parse_args().config)
