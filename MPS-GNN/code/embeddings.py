import torch
from torch import Tensor
from torch import nn, Tensor
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device
                                       ] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )
        # print(self.model)

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))
    
class BertTextEmbedding:
    """
    A simple wrapper around SentenceTransformer to generate embeddings 
    for a given list of sentences (strings).
    """
    def __init__(self, device: Optional[torch.device] = None):
        # You can change the model name to any of the best-performing SentenceTransformer models
        # For example: 'all-mpnet-base-v2' or 'all-MiniLM-L6-v2'
        model_name = "all-mpnet-base-v2"
        
        if device is None:
            # Use GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, sentences: List[str]) -> torch.Tensor:
        """
        Given a list of sentences (strings), return their embeddings as a torch.Tensor.
        """
        # sentence_transformers.encode() returns a NumPy array, so we convert it to a torch.Tensor.
        embeddings = self.model.encode(sentences)
        return torch.from_numpy(embeddings)