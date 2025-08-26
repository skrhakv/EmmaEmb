import logging
import os
from typing import List

import numpy as np
import torch

from plm_embeddings.embedding_handler import EmbeddingHandler
from plm_embeddings.embedding_model_metadata_handler import (
    EmbeddingModelMetadataHandler,
)
from plm_embeddings.utils import write_fasta

from esm import (
    Alphabet,
    FastaBatchedDataset,
    ProteinBertModel,
    pretrained,
    MSATransformer,
)


# Script kindly provided by Facebook AI Research (FAIR)
# https://github.com/facebookresearch/esm/blob/main/scripts/extract.py


class EsmFair(EmbeddingHandler):
    def __init__(
        self,
        logger: logging.Logger = None,
        no_gpu: bool = False,
    ):
        super().__init__(logger, no_gpu)

    def get_embedding(
        self,
        model_id: str,
        protein_sequences: dict,
        output_dir: str,
        toks_per_batch: int = 4096,
        layer=-1,
        include: List[str] = ["mean"],
        truncation_seq_length: int = 1022,
        model_dir: str=None,
        per_protein: bool = True
    ):
        """script to extract representations from an ESM model

        Args:
            model_id (str): PyTorch model file OR name of pretrained
                model to download (see README for models)
            protein_sequences (dict): protein sequences to embed
            output_dir (str): output directory for extracted representations
            toks_per_batch (int, optional): maximum batch size.
                Defaults to 4096.
            repr_layers (List[int], optional): layers indices from which to
                extract representations (0 to num_layers, inclusive).
                Defaults to [-1].
            include (List[str], optional): which representations to include.
            truncation_seq_length (int, optional): truncate sequences
                longer than the given value. Defaults to 1022.
            model_dir: Directory for model output. Should be a non-empty string
                or missing.

        Returns:
            _type_: _description_
        """
        os.environ['TORCH_HOME'] = model_dir
        
        repr_layers = [layer]

        # validate model parameters
        model_metadata = EmbeddingModelMetadataHandler()
        model_metadata.validate_model_id(model_id)
        model_metadata.validate_repr_layers(
            model_id=model_id, repr_layers=repr_layers
        )
        model_metadata.validate_sequence_length(
            model_id=model_id, sequence_length=truncation_seq_length
        )

        # through error if model is not loaded
        self.model, alphabet = pretrained.load_model_and_alphabet(model_id)
        cached_model_path = os.path.join(os.getenv("TORCH_HOME", "~/.cache/torch/hub"), "checkpoints", model_id)
        print(f"Model cached at: {os.path.expanduser(cached_model_path)}")
        self.model.eval()

        if isinstance(self.model, MSATransformer):
            raise ValueError(
                "This script currently does not handle models with MSA input."
            )

        if self.device.type == "cuda":
            self.model = self.model.cuda()
            print("Transferred model to GPU")

        path_fasta_file = output_dir / "temp.fasta"
        write_fasta(sequences=protein_sequences, file_path=path_fasta_file)

        dataset = FastaBatchedDataset.from_file(path_fasta_file)
        batches = dataset.get_batch_indices(
            toks_per_batch, extra_toks_per_seq=1
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=alphabet.get_batch_converter(truncation_seq_length),
            batch_sampler=batches,
        )
        print(f"Read {path_fasta_file} with {len(dataset)} sequences")

        output_dir.mkdir(parents=True, exist_ok=True)
        return_contacts = "contacts" in include

        assert all(
            -(self.model.num_layers + 1) <= i <= self.model.num_layers
            for i in repr_layers
        )
        repr_layers = [
            (i + self.model.num_layers + 1) % (self.model.num_layers + 1)
            for i in repr_layers
        ]

        with torch.no_grad():
            for batch_idx, (labels, strs, toks) in enumerate(data_loader):
                print(
                    f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
                )
                if self.device.type == "cuda":
                    toks = toks.to(device="cuda", non_blocking=True)

                out = self.model(
                    toks,
                    repr_layers=repr_layers,
                    return_contacts=return_contacts,
                )

                representations = {
                    layer: t.to(device="cpu")
                    for layer, t in out["representations"].items()
                }
                if return_contacts:
                    contacts = out["contacts"].to(device="cpu")

                for i, label in enumerate(labels):
                    output_file = output_dir / f"{label}.npy"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    result = {"label": label}
                    truncate_len = min(truncation_seq_length, len(strs[i]))
                    # Call clone on tensors to ensure tensors are not views
                    # into a larger representation
                    # See https://github.com/pytorch/pytorch/issues/1995
                    if per_protein:
                        result["representations"] = {
                            layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                            for layer, t in representations.items()
                        }
                    else:
                        result["representations"] = {
                            layer: t[i, 1 : truncate_len + 1].clone()
                            for layer, t in representations.items()
                        }
                    if return_contacts:
                        result["contacts"] = contacts[
                            i, :truncate_len, :truncate_len
                        ].clone()

                    # save result['mean_representations'] to output_file as a np array
                    embedding = (
                        result["representations"][layer].cpu().numpy()
                    )
                    np.save(
                        output_file,
                        embedding,
                    )

        os.remove(path_fasta_file)
        return result
