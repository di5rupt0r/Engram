"""Redis client with hybrid search and RRF implementation for Engram."""

import redis
from typing import List, Dict, Optional, Any
import json
import uuid
import time
import logging

import numpy as np

logger = logging.getLogger(__name__)


class EngramRedisClient:
    """Redis client with specialized operations for Engram knowledge graph."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, password: Optional[str] = None):
        """Initialize Redis client with connection pooling and retries."""
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            password=password,
            decode_responses=True,
            retry_on_timeout=True,
            health_check_interval=30
        )
        self.client = redis.Redis(connection_pool=self.pool)
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test Redis connection."""
        try:
            self.client.ping()
            logger.info("Redis connection established via pool")
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
                "links": {"inbound": [], "outbound": []}
            }
            
            self.client.json().set(f"node:{node_id}", "$", node_data)
            return True
        except Exception as e:
            logger.error(f"Failed to store node {node_id}: {e}")
            return False
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a node by ID."""
        try:
            return self.client.json().get(f"node:{node_id}")
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None
    
    def update_manifest(self, domain: str, entity_id: str) -> bool:
        """Update the domain-partitioned manifest."""
        try:
            # 1. Update the global domains list (Set)
            self.client.sadd("manifest:domains", domain)
            
            # 2. Update the domain-specific entities (Set)
            self.client.sadd(f"manifest:domain:{domain}", entity_id)
            
            # 3. Update the global sync manifest (Legacy-compatibility/Summary)
            manifest_summary = {
                "last_updated": time.time(),
                "last_entity": entity_id,
                "last_domain": domain
            }
            self.client.json().set("node:system:manifest", "$", manifest_summary)
            return True
        except Exception as e:
            logger.error(f"Failed to update manifest for {domain}/{entity_id}: {e}")
            return False
    
    def get_manifest(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve the system manifest (partitioned)."""
        try:
            if domain:
                entities = self.client.smembers(f"manifest:domain:{domain}")
                return {
                    "domain": domain,
                    "entities": sorted(list(entities)),
                    "count": len(entities)
                }
            
            domains = self.client.smembers("manifest:domains")
            return {
                "domains": sorted(list(domains)),
                "total_domains": len(domains),
                "updated_at": time.time()
            }
        except Exception as e:
            logger.error(f"Failed to get manifest: {e}")
            return {}
    
    def search_hybrid(self, query: str, query_embedding: List[float], 
                      domain_filter: Optional[List[str]] = None,
                      type_filter: Optional[List[str]] = None,
                      limit: int = 5) -> List[Dict[str, Any]]:
        """Hybrid search using RRF logic across Text and Vector results."""
        try:
            # Construct text query part
            text_query = query if query and query != "*" else "*"
            
            # Construct filters
            filters = []
            if domain_filter:
                filters.append(f"@domain:{{{ '|'.join(domain_filter) }}}")
            if type_filter:
                filters.append(f"@type:{{{ '|'.join(type_filter) }}}")
            
            filter_str = " ".join(filters)
            
            # Execute Vector Search
            # FT.SEARCH engram_nodes "(@domain:{...} @type:{...})=>[KNN 10 @embedding $blob AS score]" PARAMS 2 blob <blob> DIALECT 2
            vector_query = f"({filter_str})=>[KNN {limit*2} @embedding $blob AS vector_score]" if filter_str else f"*=>[KNN {limit*2} @embedding $blob AS vector_score]"
            
            blob = np.array(query_embedding, dtype=np.float32).tobytes()
            
            v_results = self.client.execute_command(
                "FT.SEARCH", "engram_nodes", vector_query,
                "PARAMS", "2", "blob", blob,
                "SORTBY", "vector_score", "ASC",
                "LIMIT", "0", str(limit * 2),
                "DIALECT", "2"
            )
            
            # Execute Text Search (BM25)
            t_query = f"({filter_str}) {text_query}" if filter_str else text_query
            t_results = self.client.execute_command(
                "FT.SEARCH", "engram_nodes", t_query,
                "LIMIT", "0", str(limit * 2),
                "DIALECT", "2"
            )
            
            # Reciprocal Rank Fusion (RRF)
            # RRF(d) = sum( 1 / (k + rank(i, d)) )
            k = 60
            scores = {} # doc_id -> score
            
            def process_results(results, weight=1.0):
                if not results or len(results) <= 1: return
                # First element is count, then pairs of (id, fields)
                for i in range(1, len(results), 2):
                    doc_id = results[i].replace("node:", "")
                    rank = (i // 2) + 1
                    scores[doc_id] = scores.get(doc_id, 0) + weight * (1.0 / (k + rank))

            process_results(v_results)
            process_results(t_results)
            
            # Sort and limit
            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
            
            return [{"id": doc_id, "rrf_score": score} for doc_id, score in sorted_docs]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def search_exact(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Pure BM25 text search."""
        try:
            results = self.client.execute_command(
                "FT.SEARCH", "engram_nodes", query,
                "LIMIT", "0", str(limit),
                "DIALECT", "2"
            )
            
            output = []
            if results and len(results) > 1:
                for i in range(1, len(results), 2):
                    doc_id = results[i].replace("node:", "")
                    # Extract content/domain from fields if needed, but for now just ID
                    output.append({"id": doc_id})
            return output
        except Exception as e:
            logger.error(f"Exact search failed: {e}")
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
                    return False
            return True
        except Exception as e:
            logger.error(f"Patch application failed: {e}")
            return False
