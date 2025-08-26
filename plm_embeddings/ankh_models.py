import logging
from typing import List
import os

import numpy as np


import torch
from torch.utils.data import DataLoader, Dataset
import ankh

from plm_embeddings.embedding_handler import EmbeddingHandler
from plm_embeddings.embedding_model_metadata_handler import (
    EmbeddingModelMetadataHandler,
)

class ProteinDataset(Dataset):
    """Dataset for protein sequences."""
    def __init__(self, protein_sequences):
        self.protein_ids, self.sequences = zip(*protein_sequences.items())
        self.protein_ids = list(self.protein_ids)
        self.sequences = list(self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.protein_ids[idx], self.sequences[idx]
    
class Ankh(EmbeddingHandler):
    def __init__(
        self,
        logger: logging.Logger = None,
        no_gpu: bool = False,
    ):
        super().__init__(logger, no_gpu)

    def get_embedding(
        self,
        model_id,
        protein_sequences: dict,
        output_dir: str,
        batch_size: int = 16,
        include: List[str] = ["mean"],
        truncation_seq_length: int = 1022,
        layer: int = None,
        model_dir: str = "",
        per_protein: bool = True,
    ):
        """
        Extract representations from the Ankh model.

        Args:
            model: Pretrained model loaded using `ankh.load_large_model`.
            tokenizer: Tokenizer associated with the Ankh model.
            protein_sequences (List[str]): List of protein sequences.
            output_dir (str): Output directory for extracted representations.
            include (List[str], optional): Which representations to include. Defaults to ["mean"].
            truncation_seq_length (int, optional): Maximum sequence length for truncation. Defaults to 1022.

        Returns:
            dict: Dictionary with embeddings for each sequence.
        """
        
        if layer:
            raise ValueError("Ankh embeddings only \
                implemented for the last layer.")
    
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        if model_id == "ankh_base":
            model, tokenizer = ankh.load_large_model()
        elif model_id == "ankh_large":
            model, tokenizer = ankh.load_large_model()
        else:
            raise ValueError(f"Unsupported model_id: {model_id}")
        
        model.eval()
        model.to(self.device)
        
        dataset = ProteinDataset(protein_sequences)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        for batch in dataloader:
            # Unpack the batch into protein_ids and sequences
            batch_protein_ids, batch_sequences = batch
            
            # Convert batch_protein_ids and batch_sequences to dict
            # with protein_id as key and sequence as value
            batch_sequences_dict = {pid: seq for pid, seq in zip(batch_protein_ids, batch_sequences)}
            
            outputs = tokenizer.batch_encode_plus(
                batch_sequences,  
                add_special_tokens=True,
                padding=True,
                truncation=True,
                is_split_into_words=False,  
                return_tensors="pt",
                max_length=truncation_seq_length  
            )

            input_ids = outputs["input_ids"].to(self.device)
            attention_mask = outputs["attention_mask"].to(self.device)
            
            with torch.no_grad():
                embeddings = model(input_ids=input_ids, attention_mask=attention_mask)

            # Save embeddings
            for pid, embedding in zip(batch_protein_ids, embeddings.last_hidden_state):
                output_file = os.path.join(output_dir, f"{pid}.npy")
                
                s_len = len(batch_sequences_dict[pid])
                
                # slice off padding and special token
                embedding = embedding[:s_len]
                
                if per_protein:
                    embedding_mean = embedding.mean(dim=0).cpu().numpy()  
                    np.save(output_file, embedding_mean)
                else:
                    # Save the entire embedding for each protein
                    embedding = embedding.cpu().numpy()
                    np.save(output_file, embedding)
                
                