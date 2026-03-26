"""Tests for Redis index setup following TDD principles."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from engram.index.setup import create_index, verify_index, setup_redis_index


class TestIndexSetup:
    """Test cases for Redis index setup functionality."""
    
    def test_create_index_success(self):
        """Test successful index creation."""
        with patch('engram.index.setup.redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            mock_ft = Mock()
            mock_instance.ft.return_value = mock_ft
            
            result = create_index(mock_instance)
            
            assert result is True
            mock_instance.ft.assert_called_with("engram_nodes")
            mock_ft.create_index.assert_called_once()
    
    def test_create_index_with_drop_existing(self):
        """Test index creation with dropping existing index."""
        with patch('engram.index.setup.redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            mock_ft = Mock()
            mock_instance.ft.return_value = mock_ft
            
            result = create_index(mock_instance, drop_existing=True)
            
            assert result is True
            mock_ft.dropindex.assert_called_once()
            mock_ft.create_index.assert_called_once()
    
    def test_create_index_failure(self):
        """Test index creation failure."""
        with patch('engram.index.setup.redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            mock_ft = Mock()
            mock_instance.ft.return_value = mock_ft
            mock_ft.create_index.side_effect = Exception("Creation failed")
            
            result = create_index(mock_instance)
            
            assert result is False
    
    def test_verify_index_success(self):
        """Test successful index verification."""
        with patch('engram.index.setup.redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            mock_ft = Mock()
            mock_instance.ft.return_value = mock_ft
            
            # Mock index info
            mock_info = {
                "attributes": [
                    {"name": "domain"},
                    {"name": "type"},
                    {"name": "content"},
                    {"name": "embedding", "vector": {"DIM": 384}},
                    {"name": "created_at"}
                ]
            }
            mock_ft.info.return_value = mock_info
            
            result = verify_index(mock_instance)
            
            assert result is True
            mock_instance.ft.assert_called_with("engram_nodes")
            mock_ft.info.assert_called_once()
    
    def test_verify_index_missing_field(self):
        """Test index verification with missing field."""
        with patch('engram.index.setup.redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            mock_ft = Mock()
            mock_instance.ft.return_value = mock_ft
            
            # Mock index info with missing field
            mock_info = {
                "attributes": [
                    {"name": "domain"},
                    {"name": "type"},
                    {"name": "content"}
                ]
            }
            mock_ft.info.return_value = mock_info
            
            result = verify_index(mock_instance)
            
            assert result is False
    
    def test_verify_index_wrong_dimension(self):
        """Test index verification with wrong vector dimension."""
        with patch('engram.index.setup.redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            mock_ft = Mock()
            mock_instance.ft.return_value = mock_ft
            
            # Mock index info with wrong dimension
            mock_info = {
                "attributes": [
                    {"name": "domain"},
                    {"name": "type"},
                    {"name": "content"},
                    {"name": "embedding", "vector": {"DIM": 512}},
                    {"name": "created_at"}
                ]
            }
            mock_ft.info.return_value = mock_info
            
            result = verify_index(mock_instance)
            
            assert result is False
    
    def test_verify_index_failure(self):
        """Test index verification failure."""
        with patch('engram.index.setup.redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            mock_ft = Mock()
            mock_instance.ft.return_value = mock_ft
            mock_ft.info.side_effect = Exception("Info failed")
            
            result = verify_index(mock_instance)
            
            assert result is False
    
    def test_setup_redis_index_success(self):
        """Test complete Redis setup success."""
        with patch('engram.index.setup.redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            mock_instance.ping.return_value = True
            
            with patch('engram.index.setup.verify_index') as mock_verify:
                mock_verify.return_value = True
                
                result = setup_redis_index()
                
                assert result is True
                mock_instance.ping.assert_called_once()
                mock_verify.assert_called_once_with(mock_instance)
    
    def test_setup_redis_index_connection_failure(self):
        """Test Redis setup with connection failure."""
        with patch('engram.index.setup.redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            mock_instance.ping.side_effect = Exception("Connection failed")
            
            result = setup_redis_index()
            
            assert result is False
    
    def test_setup_redis_index_creates_new_index(self):
        """Test Redis setup creates new index when none exists."""
        with patch('engram.index.setup.redis.Redis') as mock_redis:
            mock_instance = Mock()
            mock_redis.return_value = mock_instance
            mock_instance.ping.return_value = True
            
            with patch('engram.index.setup.verify_index') as mock_verify:
                with patch('engram.index.setup.create_index') as mock_create:
                    mock_verify.return_value = False
                    mock_create.return_value = True
                    
                    result = setup_redis_index()
                    
                    assert result is True
                    mock_verify.assert_called_once_with(mock_instance)
                    mock_create.assert_called_once_with(mock_instance, drop_existing=False)
