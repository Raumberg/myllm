import os
import sys
import toml
from typing import List, Optional, Tuple, Type
import dataclasses
from dataclasses import dataclass
from transformers import HfArgumentParser

class ArgParser(HfArgumentParser):
    def parse_toml(self, toml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a TOML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            toml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the TOML file and the command line
        """
        arg_list = self.parse_toml_file(os.path.abspath(toml_arg))

        outputs = []
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args} if other_args else {}
        used_args = {}

        for data_toml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_toml) if f.init}
            inputs = {k: v for k, v in vars(data_toml).items() if k in keys}
            for arg, val in other_args.items():
                if arg in keys:
                    base_type = data_toml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    if base_type is bool:
                        inputs[arg] = val.lower() in ["true", "1"]

                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse_toml_file(self, toml_path: str) -> List[dataclass]:
        """
        Load a TOML file and convert it into a list of dataclasses, extracting only the relevant fields.

        Args:
            toml_path (`str`): The path to the TOML file.

        Returns:
            [`List[dataclass]`]: A list of dataclasses populated with the values from the TOML file.
        """
        toml_data = toml.load(toml_path)

        sections = [
            'datasets',
            'run',
            'training',
            'model',
            'tokenizer',
            'lora',

            'grpo',
            'fusion'
        ]

        instances = []
        
        # Iterate through each dataclass type
        for dataclass_type in self.dataclass_types:
            # Create a dictionary to hold the filtered data for the current dataclass
            filtered_data = {}
            
            # Check if the dataclass has fields that match the keys in the TOML data
            for field in dataclass_type.__dataclass_fields__:
                # Check each section in the TOML data
                for section in toml_data:
                    if section in sections:
                        if field in toml_data[section]:
                            filtered_data[field] = toml_data[section][field]
                    else:
                        raise ValueError('Section is not available. Avaliable sections: {sections}')

            # If there is any filtered data, create an instance of the dataclass
            if filtered_data:
                print(f"\n\nCreating instance of {dataclass_type.__name__} with data: {filtered_data}\n\n") 
                instance = dataclass_type(**filtered_data)
                instances.append(instance)

        return instances


    def parse(self) -> Type[dataclass] | Tuple[Type[dataclass]]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".toml"):
            output = self.parse_toml_file(os.path.abspath(sys.argv[1]))
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".toml"):
            output = self.parse_toml(os.path.abspath(sys.argv[1]), sys.argv[2:])
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output