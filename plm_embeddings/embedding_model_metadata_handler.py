import json

from typing import List

path_model_parameters = "plm_embeddings/embedding_model_metadata.json"


class EmbeddingModelMetadataHandler:
    """
    Class to handle metadata for embedding models.

    """

    def __init__(self, path_model_parameters: str = path_model_parameters):
        self.path_model_parameters = path_model_parameters
        self.model_parameters = self.load_model_parameters()

    def load_model_parameters(self):
        with open(self.path_model_parameters, "r") as file:
            model_parameters = json.load(file)
        return model_parameters

    def get_model_names(self):
        model_names = list(
            set(
                [
                    model_parameter["model_name"]
                    for model_parameter in self.model_parameters.values()
                ]
            )
        )
        return model_names

    def get_model_ids(self):
        model_ids = list(self.model_parameters.keys())
        return model_ids

    def get_model_information_per_model_name(self, model_name):
        models_per_model_name = {
            model_id: parameter
            for model_id, parameter in self.model_parameters.items()
            if parameter["model_name"] == model_name
        }
        return models_per_model_name

    def get_metadata_per_model_id(self, model_name: str):
        model_parameters = self.model_parameters[model_name]
        return model_parameters

    def get_model_handler_per_model_id(self, model_name: str):
        model_parameters = self.get_metadata_per_model_id(model_name)
        model_handler = model_parameters["embedding_handler"]
        return model_handler

    def get_last_layer_per_model_id(self, model_name: str):
        # check if model has a last layer
        if "layers" in self.get_metadata_per_model_id(model_name):
            max_layers = self.get_metadata_per_model_id(model_name)["layers"]
        else:
            max_layers = None

        return max_layers

    def get_max_seq_lengtg_per_model(self, model_name: str):
        max_seq_length = self.get_metadata_per_model_id(model_name)[
            "max_sequence_length"
        ]
        return max_seq_length

    def validate_model_id(self, model_id):
        if model_id not in self.model_parameters:
            raise ValueError(
                f"Model ID {model_id} not found in model parameters."
            )

    def validate_repr_layers(self, model_id: str, repr_layers: List[int]):
        model_parameters = self.get_metadata_per_model_id(model_id)
        # test whether the model accepts a representation layer parameter
        if "layers" not in model_parameters and repr_layers == [-1]:
            pass
        elif "layers" not in model_parameters and repr_layers != [-1]:
            raise ValueError(
                f"For model {model_id}, only the last layer (-1) can be used \
                    as representation layer."
            )
        else:
            max_repr_layers = model_parameters["layers"]
            # needs to be in range or - repr_layers to repr_layers
            if not all(
                [
                    layer in range(-max_repr_layers, max_repr_layers + 1)
                    for layer in repr_layers
                ]
            ):
                raise ValueError(
                    f"Invalid representation layers. \
                        Must be in range -{max_repr_layers} \
                            to {max_repr_layers}."
                )

    def validate_sequence_length(self, model_id: str, sequence_length: int):
        model_parameters = self.get_metadata_per_model_id(model_id)
        max_sequence_length = model_parameters["max_sequence_length"]
        if sequence_length > max_sequence_length:
            raise ValueError(
                f"Invalid sequence length. \
                    Must be less than or equal to {max_sequence_length}."
            )
        return True
