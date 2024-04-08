from typing import Dict, Any
import peft


class PeftHelper:
    def __init__(self, peft_arguments: Dict[str, Any]) -> None:
        self.peft_arguments = peft_arguments
        self.peft_config = self._construct_config(self.peft_arguments)
    
    @staticmethod
    def _construct_config(peft_arguments: Dict[str, Any]):
        module_config = getattr(peft, peft_arguments["class_config"]["class_name"])
        return module_config(
            **peft_arguments["class_config"]["class_arguments"]
        )
    
    
    def get_model(self, model, prepare_for_kbit: bool = False):
        if prepare_for_kbit:
            print("Preparing for kbit")
            model = peft.prepare_model_for_kbit_training(model)
        model = peft.get_peft_model(model, self.peft_config)
        model.print_trainable_parameters()
        return model
    
    def get_model(self, model, prepare_for_kbit: bool = False):
        if prepare_for_kbit:
            print("Preparing for kbit")
            model = peft.prepare_model_for_kbit_training(model)
        model = peft.get_peft_model(model, self.peft_config)
        model.print_trainable_parameters()
        return model
