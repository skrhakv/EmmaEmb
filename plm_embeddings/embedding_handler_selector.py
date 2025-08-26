import logging

from plm_embeddings.embedding_model_metadata_handler import (
    EmbeddingModelMetadataHandler,
)

def select_embedding_handler(
    model_id: str, logger: logging.Logger, no_gpu: bool = False
):
    """
    Factory function to select the appropriate embedding handler
    based on model name.

    Args:
        model_id (str): Name of the embedding model to be used.

    Returns:
        EmbeddingHandler: An instance of the appropriate embedding
            handler class.
    """
    model_metadata = EmbeddingModelMetadataHandler()
    model_metadata.validate_model_id(model_id)
    model_handler = model_metadata.get_model_handler_per_model_id(model_id)
    if model_handler == "T5":
        from plm_embeddings.t5 import T5
        return T5(logger=logger, no_gpu=no_gpu)
    elif model_handler == "EsmFair":
        from plm_embeddings.esm_fair import EsmFair
        return EsmFair(logger=logger, no_gpu=no_gpu)
    elif model_handler == "Ankh":
        from plm_embeddings.ankh_models import Ankh
        return Ankh(logger=logger, no_gpu=no_gpu)
    elif model_handler == "Esm3":
        from plm_embeddings.esm3 import Esm3
        return Esm3(logger=logger, no_gpu=no_gpu)
