import os

PATH = "/scratch/protein_models/"
os.environ['HF_HOME'] = PATH

import torch
from abc import ABC, abstractmethod
import logging


# Define the abstract base class for embedding handlers
class EmbeddingHandler(ABC):
    def __init__(self, logger: logging.Logger = None, no_gpu: bool = False):
        """
        Base class for embedding handlers.

        Args:
            no_gpu (bool): Flag to disable GPU usage.
        """
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            print("Warning! New logger for model.")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not no_gpu else "cpu"
        )
        logging.info(f"Using device: {self.device}")
        print(f"Device: {self.device}")

    @abstractmethod
    def get_embedding(
        self, protein_sequences: dict, model_id, output_dir: str, layer: int, model_dir: str, per_protein: bool
    ):
        pass

    def check_device(self):
        """
        Logs the device being used for computations.
        """
        if self.device.type == "cuda":
            self.logger.info(f"Using GPU for {self.model_id} embeddings.")
        else:
            self.logger.info(f"Using CPU for {self.model_id} embeddings.")
