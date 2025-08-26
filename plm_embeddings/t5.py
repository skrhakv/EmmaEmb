import time
import logging

import torch
import numpy as np

from torch.utils.data import Dataset
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm

from plm_embeddings.embedding_handler import EmbeddingHandler


class ProteinDataset(Dataset):
    def __init__(self, sequence_dict):
        self.protein_ids, self.sequences = zip(*sequence_dict.items())

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.protein_ids[idx], self.sequences[idx]


def get_T5_model(
    model_dir,
    transformer_link: str,
    device: torch.device,
):
    """Load a T5 model from Huggingface Transformers"""
    print("Loading: {}".format(transformer_link))
    if model_dir is not None:
        print("##########################")
        print("Loading cached model from: {}".format(model_dir))
        print("##########################")
    else:
        print(
            f"Loading model into '{model_dir}'"
        )
    model = T5EncoderModel.from_pretrained(
        transformer_link, cache_dir=model_dir
    )
    # only cast to full-precision if no GPU is available
    if device == torch.device("cpu"):
        print("Casting model to full precision for running on CPU ...")
        model.to(torch.float32)

    model = model.to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False, cache_dir=model_dir)
    return model, vocab


def process_protein_sequences(
    protein_sequences: dict, truncation_seq_length: int
):
    """Sort and prepare protein sequences for batching"""
    sorted_sequences = sorted(
        protein_sequences.items(),
        key=lambda kv: len(protein_sequences[kv[0]]),
        reverse=True,
    )
    for pdb_id, seq in sorted_sequences:
        seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
        len_seq = len(seq)
        seq = " ".join(list(seq))  # Add spaces between characters
        yield pdb_id, seq, len_seq


def write_embedding_to_file(
    identifier: str, embedding: np.array, output_dir: str, extension: str
):
    """Save the embedding to a file"""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{identifier}.{extension}"
    if extension == "npy":
        np.save(file_path, embedding)
    else:
        raise ValueError(f"Extension {extension} not supported")
    logging.info(f"Embedded protein {identifier} to {file_path}")


# Script kindly provided by Rost Lab and adapted to the EMMA framework
# https://github.com/agemagician/ProtTrans/blob/master/Embedding/prott5_embedder.py
class T5(EmbeddingHandler):
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
        truncation_seq_length: int,
        model_dir: str,
        batch_size: int = 8,
        extension: str = "npy",
        layer: int = -1,
        per_protein: bool = True
    ):
        max_batch = batch_size  # max number of sequences per single batch
        max_residues = 4000  # number of cummulative residues per batch
        max_seq_len = truncation_seq_length

        emb_dict = dict()
        processed_sequences = process_protein_sequences(
            protein_sequences, max_seq_len
        )

        model, vocab = get_T5_model(
            model_dir=model_dir,
            transformer_link=model_id,
            device=self.device,
        )

        start = time.time()
        batch = list()
        for seq_idx, (pdb_id, seq, seq_len) in tqdm(enumerate(processed_sequences), 
                                                    total=len(protein_sequences), 
                                                    desc="Processing sequences"):
            batch.append((pdb_id, seq, seq_len))

            # count residues in current batch and add the last sequence length
            # to avoid that batches with (n_res_batch > max_residues) get
            # processed
            n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len
            if (
                len(batch) >= max_batch
                or n_res_batch >= max_residues
                or seq_idx == len(protein_sequences) -1  # last sequence
                or seq_len > max_seq_len
            ):
                pdb_ids, seqs, seq_lens = zip(*batch)
                batch = list()

                token_encoding = vocab.batch_encode_plus(
                    seqs, add_special_tokens=True, padding="longest"
                )
                input_ids = torch.tensor(token_encoding["input_ids"]).to(
                    self.device
                )
                attention_mask = torch.tensor(
                    token_encoding["attention_mask"]
                ).to(self.device)

                try:
                    with torch.no_grad():
                        embedding_repr = model(
                            input_ids, attention_mask=attention_mask
                        )
                except RuntimeError:
                    logging.info(
                        "RuntimeError during embedding for {} (L={}). \
                            Try lowering batch size. ".format(
                            pdb_id, seq_len
                        )
                        + "If single sequence processing does not work, you \
                            need more vRAM to process your protein."
                    )

                # batch_size x seq_len x embedding_dim
                # extra token is added at the end of the seq
                for batch_idx, identifier in enumerate(pdb_ids):
                    s_len = seq_lens[batch_idx]
                    # slice off padded/special tokens
                    emb = embedding_repr.last_hidden_state[batch_idx, :s_len]

                    if per_protein:
                        emb = emb.mean(dim=0)                   
                    if len(emb_dict) == 0:
                        logging.info(
                            "Embedded prrotein {} with length {} to emb. \
                            of shape: {}".format(
                                identifier, s_len, emb.shape
                            )
                        )
                        
                    emb_dict[identifier] = emb.detach().cpu().numpy().squeeze()

                    write_embedding_to_file(
                        identifier=identifier,
                        embedding=emb_dict[identifier],
                        output_dir=output_dir,
                        extension=extension,
                    )
        end = time.time()
        logging.info(f"Finished embedding in {end - start:.2f} seconds")
