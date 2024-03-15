import yaml
import Dict, Any
from pathlib import Path
from argparse import ArgumentParser
import transformers
from transformers import Trainer, AutoModel, AutoTokenizer
from modeling.peft_training import PeftHelper


# TODO: implement w. jsonargparse & custom parsers for descript training arguments later
# from jsonargparse import CLI


def load_model_and_tokenizer(model_name_or_path:str, base_class: str):
    model_class = getattr(transformers, base_class)
    model = model_class.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer

def modeling_sanity_checks():
    return True


def parse_config(config):
    # TODO: once we have parsers for diff classes
    return config


def main(config: Path):
    with open(config, "r") as reader:
        config = yaml.full_load(reader)
    
    modeling_sanity_checks()  # validate args

    model_arguments = config["model_arguments"]

    # load base model
    model_name_or_path = model_arguments["model_name_or_path"]
    base_class = model_arguments["base_class"]

    model, tokenizer = load_model_and_tokenizer(model_name_or_path, base_class)

    # need to handle peft?
    if peft_arguments := config.get("peft_arguments", False):
        peft_helper = PeftHelper(peft_arguments)
        model = peft_helper.model

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer
    )
    


    trainer.train()

    pass

if __name__ == "__main__":
    # CLI(main)
    parser = ArgumentParser()
    parser.add_argument("config", type=Path)
    main(parser.parse_args.config)
