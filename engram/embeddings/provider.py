"""Embedding provider using FastEmbed for CPU-optimized vector generation."""

import numpy as np
from typing import List, Optional
import logging
from fastembed import TextEmbedding
import time

logger = logging.getLogger(__name__)

# Global embedding model instance
_embedding_model: Optional[TextEmbedding] = None


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> TextEmbedding:
    """Get or create the embedding model instance."""
    global _embedding_model
    
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {model_name}")
        start_time = time.time()
        
        _embedding_model = TextEmbedding(
            model_name=model_name,
            cache_dir="/tmp/fastembed_cache",
        )
        
        load_time = time.time() - start_time
        logger.info(f"Embedding model loaded in {load_time:.2f}s")
    
    return _embedding_model


def generate_embedding(text: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[float]:
    """Generate embedding for a single text."""
    if not text or not text.strip():
        raise ValueError("Cannot generate embedding for empty text")
    
    model = get_embedding_model(model_name)
    
    try:
        embedding_generator = model.embed([text])
        embedding = next(embedding_generator)
        
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        else:
            return list(embedding)
            
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise


def generate_embeddings_batch(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[List[float]]:
    """Generate embeddings for multiple texts efficiently."""
    if not texts:
        return []
    
    valid_texts = [text for text in texts if text and text.strip()]
    if not valid_texts:
        return []
    
    model = get_embedding_model(model_name)
    
    try:
        embedding_generator = model.embed(valid_texts)
        embeddings = list(embedding_generator)
        
        result = []
        for embedding in embeddings:
            if isinstance(embedding, np.ndarray):
                result.append(embedding.tolist())
            else:
                result.append(list(embedding))
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate batch embeddings: {e}")
        raise


def get_embedding_dimension() -> int:
    """Get the dimension of the embedding vectors."""
    return 384


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension")
    
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(y * y for y in b) ** 0.5
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)
