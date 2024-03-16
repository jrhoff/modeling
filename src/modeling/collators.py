import torch
from typing import Any, Dict, List, Optional


class CollatorForCausalCompletions:

    def __init__(self, tokenizer, label_pad_idx=-100, max_sequence_length: Optional[int] = None) -> None:
        self.input_pad_idx = tokenizer.pad_token_id
        self.label_pad_idx = label_pad_idx
        self.max_sequence_length = max_sequence_length
        

    @staticmethod
    def pad_right(x: torch.Tensor, pad_id: int, max_length: int):
            # pad right based on the longest sequence
            n = max_length - len(x)
            return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    @staticmethod
    def pad_inputs_and_labels(examples: List[Dict], input_pad_idx, label_pad_idx, max_sequence_length):

        input_ids = [e["input_ids"].type(torch.int64) for e in examples]
        labels = [e["labels"].type(torch.int64) for e in examples]

        # find max length in batch, we pad dynamically
        max_length = max(len(s) for s in input_ids)


        x = torch.stack(
             [CollatorForCausalCompletions.pad_right(x, pad_id=input_pad_idx, max_length=max_length) for x in input_ids]
             )
        y = torch.stack(
             [CollatorForCausalCompletions.pad_right(x, pad_id=label_pad_idx, max_length=max_length) for x in labels]
             )

        # Truncate if needed
        if max_sequence_length is not None:
            x = x[:, :max_sequence_length]
            y = y[:, :max_sequence_length]
        return x, y


    def __call__(self, examples: List[Dict[str, Any]]) -> Any:

        input_ids, labels = CollatorForCausalCompletions.pad_inputs_and_labels(
             examples,
             self.input_pad_idx,
             self.label_pad_idx,
             self.max_sequence_length
        )

        return {
             "input_ids": input_ids,
             "labels": labels
        }