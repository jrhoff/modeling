import torch
import wandb
import re
from typing import List, Iterable
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class SampleGenerationsCallback(TrainerCallback):

    

    def __init__(self, model, tokenizer, batch_size=16, 
                 input_parse_regex="[(\\[]?(?P<index>[0-9]+)[)\\]]? ?[-:>\\.]+ ?(?P<translation>.+)", 
                 output_parse_regex="[(\\[]?(?P<index>[0-9]+)[)\\]]? ?[-:>\\.]+ ?(?P<translation>.+)") -> None:
        super().__init__()
        self.sample_location = "/home/jacob94_hoff/data/kadokawa/repaired/elyza_7b/enum=3-5_msl=512_threshold=0.95/heldout.pt"
        self.log_iter = 0
        self.model = model
        self.tokenizer = tokenizer
        self.history = []
        self.input_parse_regex = input_parse_regex
        self.output_parse_regex = output_parse_regex
        self.batch_size = batch_size
        self.metric_defined = False
        
        

    
    def log_to_wandb(self, inputs, generations):
        wandb.init()
        if not self.metric_defined:
            wandb.define_metric("fraction_parsed", summary="max")
            self.metric_defined = True

        columns = ["Prompt", "Generation", "Iteration"]
        # Method 1
        data = [ [prompt, response, self.log_iter] for (prompt, response) in zip(inputs, generations)]
        self.log_iter += 1
        self.history.append(data)

        table_data = []
        for prev_data in self.history[::-1]:  # we want descending epochs in wandb
            table_data.extend(prev_data)

        

        table = wandb.Table(data=table_data, columns=columns)
        wandb.log({f"sample_generations": table})

    @staticmethod
    def get_parsabiility(input_texts: List[str], 
                         output_texts: List[str],
                         input_regex: str,
                         output_regex: str):
        
        in_pattern = re.compile(input_regex)
        out_pattern = re.compile(output_regex)

        total_expected = 0
        total_actual = 0

        for input_text, output_text in zip(input_texts, output_texts):
            expected_number_of_outputs = list(re.finditer(in_pattern, input_text)).__len__()
            actual_number_of_outputs = list(re.finditer(out_pattern, output_text)).__len__()
            total_expected += expected_number_of_outputs
            total_actual += actual_number_of_outputs
        
        assert total_expected != 0, "Expected # outputs is 0, debug."
        
        return round(total_actual / total_expected, 3)
    
    @staticmethod
    def batch(data: Iterable, batch_size):
        if batch_size == 1:
            for ele in data:
                yield ele
        for i in range(0, len(data), batch_size):
                yield data[i:i + batch_size]

    def on_evaluate(self, args, state, control, **kwargs):
        with open(self.sample_location, "rb") as fp:
            data = torch.load(fp)
        
        model = self.model
        tokenizer = self.tokenizer
        inputs, generations = [], []
        with torch.no_grad():
            for batch in self.batch(data, self.batch_size):
                prompts = [inst["prompt"] for inst in batch]
                batch_encoded = self.tokenizer(prompts , return_tensors='pt', padding=True)
                batch_output_ids = model.generate(
                inputs=batch_encoded["input_ids"].to(model.device),
                attention_mask=batch_encoded["attention_mask"].to(model.device),
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1
            )
                for i, output_ids in enumerate(batch_output_ids):
                    corresponding_input_ids = batch_encoded["input_ids"][i]
                    output = tokenizer.decode(output_ids.tolist()[corresponding_input_ids.size(0) :], skip_special_tokens=True)
                    inputs.append(prompts[i])
                    generations.append(output)
        # log table to wandb
        self.log_to_wandb(inputs[:10], generations[:10])

        # log fraction of examples parsed
        frac_parsed = self.get_parsabiility(inputs, generations, self.input_parse_regex, self.output_parse_regex)
        wandb.log({"fraction_parsed": frac_parsed})