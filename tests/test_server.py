"""Tests for FastMCP server following TDD principles."""

import pytest
import yaml
import uuid
from unittest.mock import Mock, patch, MagicMock
from engram.server import memorize, recall, patch as patch_node, search_exact, inspect_node


class TestServerFunctions:
    """Test cases for FastMCP server functions."""
    
    def test_memorize_success(self):
        """Test successful memorize operation."""
        with patch('engram.server.get_redis_client') as mock_get_client:
            with patch('engram.server.generate_embedding') as mock_generate:
                # Arrange
                mock_client = Mock()
                mock_get_client.return_value = mock_client
                mock_generate.return_value = [0.1] * 384
                mock_client.store_node.return_value = True
                mock_client.get_manifest.return_value = None
                
                domain = "test"
                node_type = "entity"
                content = "Test content"
                metadata = {"key": "value"}
            
                # Act
                result = memorize(domain, node_type, content, metadata)
                parsed_result = yaml.safe_load(result)
            
                # Assert
                assert parsed_result["status"] == "success"
                assert "node_id" in parsed_result
                assert parsed_result["domain"] == domain
                assert parsed_result["type"] == node_type
                mock_client.store_node.assert_called_once()
                mock_client.update_manifest.assert_called_once()
    
    def test_memorize_storage_failure(self):
        """Test memorize operation with storage failure."""
        with patch('engram.server.get_redis_client') as mock_get_client:
            with patch('engram.server.generate_embedding') as mock_generate:
                # Arrange
                mock_client = Mock()
                mock_get_client.return_value = mock_client
                mock_generate.return_value = [0.1] * 384
                mock_client.store_node.return_value = False
                
                domain = "test"
                node_type = "entity"
                content = "Test content"
            
                # Act
                result = memorize(domain, node_type, content)
                parsed_result = yaml.safe_load(result)
            
                # Assert
                assert parsed_result["status"] == "error"
                assert "Failed to store node" in parsed_result["message"]
    
    def test_recall_manifest_intercept(self):
        """Test recall operation with manifest intercept."""
        with patch('engram.server.get_redis_client') as mock_get_client:
            # Arrange
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_manifest = {"domains": ["test"], "entities": ["id1"]}
            mock_client.get_manifest.return_value = mock_manifest
            
            # Act
            result = recall("manifest")
            parsed_result = yaml.safe_load(result)
            
            # Assert
            assert parsed_result == mock_manifest
    
    def test_recall_wildcard_intercept(self):
        """Test recall operation with wildcard intercept."""
        with patch('engram.server.get_redis_client') as mock_get_client:
            # Arrange
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_manifest = {"domains": ["test"], "entities": ["id1"]}
            mock_client.get_manifest.return_value = mock_manifest
            
            # Act
            result = recall("*")
            parsed_result = yaml.safe_load(result)
            
            # Assert
            assert parsed_result == mock_manifest
    
    def test_recall_normal_search(self):
        """Test normal recall operation with hybrid search."""
        with patch('engram.server.get_redis_client') as mock_get_client:
            with patch('engram.server.generate_embedding') as mock_generate:
                # Arrange
                mock_client = Mock()
                mock_get_client.return_value = mock_client
                mock_generate.return_value = [0.1] * 384
                
                search_results = [
                    {
                        "id": "test_id",
                        "rrf_score": 0.8,
                        "links": {"inbound": [], "outbound": []}
                    }
                ]
                mock_client.search_hybrid.return_value = search_results
                
                node_data = {
                    "domain": "test",
                    "type": "entity",
                    "content": "Test content",
                    "created_at": 1234567890
                }
                mock_client.get_node.return_value = node_data
                
                # Act
                result = recall("test query")
                parsed_result = yaml.safe_load(result)
                
                # Assert
                assert parsed_result["query"] == "test query"
                assert parsed_result["total_results"] == 1
                assert len(parsed_result["results"]) == 1
                assert parsed_result["results"][0]["id"] == "test_id"
    
    def test_patch_success(self):
        """Test successful patch operation."""
        with patch('engram.server.get_redis_client') as mock_get_client:
            # Arrange
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_client.get_node.return_value = {"domain": "test"}
            mock_client.apply_patch.return_value = True
            
            node_id = str(uuid.uuid4())
            operations = [{"op": "set", "path": "$.metadata.key", "value": "new_value"}]
            
            # Act
            result = patch_node(node_id, operations)
            parsed_result = yaml.safe_load(result)
            
            # Assert
            assert parsed_result["status"] == "success"
            assert parsed_result["node_id"] == node_id
            assert parsed_result["operations_applied"] == 1
    
    def test_patch_node_not_found(self):
        """Test patch operation with node not found."""
        with patch('engram.server.get_redis_client') as mock_get_client:
            # Arrange
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_client.get_node.return_value = None
            
            node_id = str(uuid.uuid4())
            operations = [{"op": "set", "path": "$.metadata.key", "value": "new_value"}]
            
            # Act
            result = patch_node(node_id, operations)
            parsed_result = yaml.safe_load(result)
            
            # Assert
            assert parsed_result["status"] == "error"
            assert "not found" in parsed_result["message"]
    
    def test_search_exact_success(self):
        """Test successful exact search operation."""
        with patch('engram.server.get_redis_client') as mock_get_client:
            # Arrange
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            search_results = [
                {
                    "id": "test_id",
                    "domain": "test",
                    "type": "entity",
                    "snippet": "Test content snippet..."
                }
            ]
            mock_client.search_exact.return_value = search_results
            
            # Act
            result = search_exact("test query")
            parsed_result = yaml.safe_load(result)
            
            # Assert
            assert parsed_result["query"] == "test query"
            assert parsed_result["search_type"] == "exact_bm25"
            assert parsed_result["total_results"] == 1
            assert len(parsed_result["results"]) == 1
    
    def test_inspect_node_success(self):
        """Test successful node inspection."""
        with patch('engram.server.get_redis_client') as mock_get_client:
            # Arrange
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            node_data = {
                "domain": "test",
                "type": "entity",
                "content": "Test content",
                "created_at": 1234567890,
                "metadata": {"key": "value"},
                "links": {"inbound": [], "outbound": []},
                "embedding": [0.1] * 384
            }
            mock_client.get_node.return_value = node_data
            
            node_id = str(uuid.uuid4())
            
            # Act
            result = inspect_node(node_id)
            parsed_result = yaml.safe_load(result)
            
            # Assert
            assert parsed_result["id"] == node_id
            assert parsed_result["domain"] == "test"
            assert parsed_result["type"] == "entity"
            assert parsed_result["content"] == "Test content"
            assert parsed_result["embedding_dimension"] == 384
    
    def test_inspect_node_not_found(self):
        """Test node inspection with node not found."""
        with patch('engram.server.get_redis_client') as mock_get_client:
            # Arrange
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            mock_client.get_node.return_value = None
            
            node_id = str(uuid.uuid4())
            
            # Act
            result = inspect_node(node_id)
            parsed_result = yaml.safe_load(result)
            
            # Assert
            assert parsed_result["status"] == "error"
            assert "not found" in parsed_result["message"]
