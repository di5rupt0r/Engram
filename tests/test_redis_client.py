"""Tests for Redis client following TDD principles."""

import pytest
import time
import uuid
from unittest.mock import Mock, patch
from engram.redis.client import EngramRedisClient


class TestEngramRedisClient:
    """Test cases for Redis client functionality."""
    
    def test_initialization_success(self):
        """Test successful client initialization."""
        with patch('engram.redis.client.redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            mock_instance.ping.return_value = True
            
            client = EngramRedisClient()
            
            mock_instance.ping.assert_called_once()
            assert client.client is not None
    
    def test_initialization_failure(self):
        """Test initialization failure handling."""
        with patch('engram.redis.client.redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            mock_instance.ping.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception, match="Connection failed"):
                EngramRedisClient()
    
    def test_store_node_success(self):
        """Test successful node storage."""
        with patch('engram.redis.client.redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            mock_instance.ping.return_value = True
            mock_instance.json.return_value.set.return_value = True
            
            client = EngramRedisClient()
            node_id = str(uuid.uuid4())
            domain = "test"
            node_type = "entity"
            content = "Test content"
            embedding = [0.1] * 384
            metadata = {"key": "value"}
        
            result = client.store_node(node_id, domain, node_type, content, embedding, metadata)
        
            assert result is True
            call_args = mock_instance.json.return_value.set.call_args
            assert call_args[0][0] == f"node:{node_id}"
            assert call_args[0][1] == "$"
            assert call_args[0][2]["domain"] == domain
    
    def test_get_node_success(self):
        """Test successful node retrieval."""
        with patch('engram.redis.client.redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            mock_instance.ping.return_value = True
            expected_data = {"domain": "test", "type": "entity", "content": "test"}
            mock_instance.json.return_value.get.return_value = expected_data
            
            client = EngramRedisClient()
            node_id = str(uuid.uuid4())
            result = client.get_node(node_id)
        
            assert result == expected_data
            mock_instance.json.return_value.get.assert_called_once_with(f"node:{node_id}")
    
    def test_update_manifest_success(self):
        """Test successful manifest update."""
        with patch('engram.redis.client.redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            mock_instance.ping.return_value = True
            mock_instance.json.return_value.set.return_value = True
            
            client = EngramRedisClient()
            domains = {"test", "docs"}
            entity_ids = {"id1", "id2"}
        
            result = client.update_manifest(domains, entity_ids)
        
            assert result is True
            call_args = mock_instance.json.return_value.set.call_args
            assert call_args[0][0] == "node:system:manifest"
            manifest_data = call_args[0][2]
            assert sorted(manifest_data["domains"]) == sorted(["docs", "test"])
            assert manifest_data["total_nodes"] == 2
