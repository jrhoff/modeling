from typing import Dict, Any
import peft


class PeftHelper:
    def __init__(self, peft_arguments: Dict[str, Any]) -> None:
        self.peft_arguments = peft_arguments
        self.config = self._construct_config(self.peft_arguments)
    
    @staticmethod
    def _construct_config(peft_arguments: Dict[str, Any]):
        module_config = getattr(peft, peft_arguments["class_config"]["class_name"])
        return module_config(
            **peft_arguments["class_config"]["class_arguments"]
        )
    
    @staticmethod
    def _get_model(model, peft_config):
        model = peft.get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model


    def train():
        
        pass