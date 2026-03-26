"""Tests for embedding provider following TDD principles."""

import pytest
import numpy as np
from engram.embeddings.provider import (
    generate_embedding, 
    generate_embeddings_batch,
    get_embedding_dimension,
    cosine_similarity
)


class TestEmbeddingProvider:
    """Test cases for embedding provider functionality."""
    
    def test_generate_single_embedding(self):
        """Test generating embedding for a single text."""
        # Arrange
        text = "This is a test sentence for embedding generation."
        
        # Act
        embedding = generate_embedding(text)
        
        # Assert
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # FastEmbed default dimension
        assert all(isinstance(x, float) for x in embedding)
        assert not any(np.isnan(x) for x in embedding)
        assert not any(np.isinf(x) for x in embedding)
    
    def test_generate_embedding_empty_text(self):
        """Test that empty text raises ValueError."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Cannot generate embedding for empty text"):
            generate_embedding("")
    
    def test_generate_embedding_whitespace_only(self):
        """Test that whitespace-only text raises ValueError."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Cannot generate embedding for empty text"):
            generate_embedding("   \n\t  ")
    
    def test_generate_batch_embeddings(self):
        """Test generating embeddings for multiple texts."""
        # Arrange
        texts = [
            "First test sentence",
            "Second test sentence", 
            "Third test sentence"
        ]
        
        # Act
        embeddings = generate_embeddings_batch(texts)
        
        # Assert
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 384
            assert all(isinstance(x, float) for x in embedding)
    
    def test_generate_batch_embeddings_empty_list(self):
        """Test batch embedding with empty list."""
        # Arrange & Act
        embeddings = generate_embeddings_batch([])
        
        # Assert
        assert embeddings == []
    
    def test_generate_batch_embeddings_with_empty_strings(self):
        """Test batch embedding filters out empty strings."""
        # Arrange
        texts = [
            "Valid sentence",
            "",
            "   ",
            "Another valid sentence"
        ]
        
        # Act
        embeddings = generate_embeddings_batch(texts)
        
        # Assert
        assert len(embeddings) == 2  # Only valid sentences
    
    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        # Act
        dimension = get_embedding_dimension()
        
        # Assert
        assert dimension == 384
    
    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity of identical vectors is 1.0."""
        # Arrange
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Act
        similarity = cosine_similarity(vector, vector)
        
        # Assert
        assert abs(similarity - 1.0) < 1e-10
    
    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors is 0.0."""
        # Arrange
        vector_a = [1.0, 0.0, 0.0, 0.0, 0.0]
        vector_b = [0.0, 1.0, 0.0, 0.0, 0.0]
        
        # Act
        similarity = cosine_similarity(vector_a, vector_b)
        
        # Assert
        assert abs(similarity - 0.0) < 1e-10
    
    def test_cosine_similarity_dimension_mismatch(self):
        """Test cosine similarity raises error for mismatched dimensions."""
        # Arrange
        vector_a = [0.1, 0.2, 0.3]
        vector_b = [0.1, 0.2]
        
        # Act & Assert
        with pytest.raises(ValueError, match="Vectors must have the same dimension"):
            cosine_similarity(vector_a, vector_b)
    
    def test_cosine_similarity_zero_vectors(self):
        """Test cosine similarity handles zero vectors."""
        # Arrange
        vector_a = [0.0, 0.0, 0.0]
        vector_b = [0.1, 0.2, 0.3]
        
        # Act
        similarity = cosine_similarity(vector_a, vector_b)
        
        # Assert
        assert similarity == 0.0
