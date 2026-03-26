"""Redis client with hybrid search and RRF implementation for Engram."""

import redis
from typing import List, Dict, Optional, Any
import json
import uuid
import time
import logging

logger = logging.getLogger(__name__)


class EngramRedisClient:
    """Redis client with specialized operations for Engram knowledge graph."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, password: Optional[str] = None):
        """Initialize Redis client."""
        self.client = redis.Redis(
            host=host,
            port=port,
            password=password,
            decode_responses=True,
        )
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test Redis connection."""
        try:
            self.client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    def store_node(self, node_id: str, domain: str, node_type: str, content: str, 
                   embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a node in RedisJSON."""
        try:
            node_data = {
                "domain": domain,
                "type": node_type,
                "content": content,
                "embedding": embedding,
                "created_at": time.time(),
                "metadata": metadata or {},
                "links": {
                    "inbound": [],
                    "outbound": []
                }
            }
            
            self.client.json().set(f"node:{node_id}", "$", node_data)
            return True
            
        except Exception as e:
            logger.error(f"Failed to store node {node_id}: {e}")
            return False
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node by ID."""
        try:
            node_data = self.client.json().get(f"node:{node_id}")
            return node_data
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None
    
    def update_manifest(self, domains: set, entity_ids: set) -> bool:
        """Update the system manifest with current domains and entities."""
        try:
            manifest_data = {
                "domains": sorted(list(domains)),
                "entities": sorted(list(entity_ids)),
                "updated_at": time.time(),
                "total_nodes": len(entity_ids)
            }
            
            self.client.json().set("node:system:manifest", "$", manifest_data)
            return True
            
        except Exception as e:
            logger.error(f"Failed to update manifest: {e}")
            return False
    
    def get_manifest(self) -> Optional[Dict[str, Any]]:
        """Retrieve the system manifest."""
        try:
            manifest_data = self.client.json().get("node:system:manifest")
            return manifest_data
        except Exception as e:
            logger.error(f"Failed to get manifest: {e}")
            return None
    
    def search_exact(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Pure BM25 text search."""
        try:
            # Placeholder implementation - will be expanded in next iteration
            return []
        except Exception as e:
            logger.error(f"Exact search failed: {e}")
            return []
    
    def search_vector(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Pure vector similarity search."""
        try:
            # Placeholder implementation - will be expanded in next iteration
            return []
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def search_hybrid(self, query: str, query_embedding: List[float], 
                     domain_filter: Optional[List[str]] = None,
                     type_filter: Optional[List[str]] = None,
                     limit: int = 5) -> List[Dict[str, Any]]:
        """Hybrid search with RRF (Reciprocal Rank Fusion)."""
        try:
            # Placeholder implementation - will be expanded in next iteration
            return []
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def apply_patch(self, node_id: str, operations: List[Dict[str, Any]]) -> bool:
        """Apply JSON patch operations to a node."""
        try:
            for operation in operations:
                op_type = operation.get("op")
                path = operation.get("path")
                value = operation.get("value")
                
                if op_type == "set":
                    self.client.json().set(f"node:{node_id}", path, value)
                elif op_type == "delete":
                    getattr(self.client.json(), 'del')(f"node:{node_id}", path)
                elif op_type == "append":
                    self.client.json().arrappend(f"node:{node_id}", path, value)
                else:
                    logger.warning(f"Unsupported patch operation: {op_type}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Patch application failed: {e}")
            return False
