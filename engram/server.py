"""FastMCP server implementation for Engram knowledge graph."""

import uuid
import time
import yaml
import logging
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP
from .redis.client import EngramRedisClient
from .embeddings.provider import generate_embedding
from .index.setup import setup_redis_index

logger = logging.getLogger(__name__)

# Create FastMCP instance
mcp = FastMCP("Engram")

# Global Redis client
_redis_client: Optional[EngramRedisClient] = None


def get_redis_client() -> EngramRedisClient:
    """Get or create Redis client instance."""
    global _redis_client
    if _redis_client is None:
        _redis_client = EngramRedisClient()
        setup_redis_index()
    return _redis_client


@mcp.tool()
def memorize(domain: str, type: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Store a node in the knowledge graph with automatic embedding generation."""
    try:
        node_id = str(uuid.uuid4())
        embedding = generate_embedding(content)
        
        redis_client = get_redis_client()
        success = redis_client.store_node(
            node_id=node_id,
            domain=domain,
            node_type=type,
            content=content,
            embedding=embedding,
            metadata=metadata
        )
        
        if success:
            current_manifest = redis_client.get_manifest()
            if current_manifest:
                domains = set(current_manifest.get("domains", []))
                entities = set(current_manifest.get("entities", []))
            else:
                domains = set()
                entities = set()
            
            domains.add(domain)
            entities.add(node_id)
            
            redis_client.update_manifest(domains, entities)
            
            result = {
                "status": "success",
                "node_id": node_id,
                "domain": domain,
                "type": type,
                "created_at": time.time()
            }
        else:
            result = {
                "status": "error",
                "message": "Failed to store node"
            }
        
        return yaml.dump(result, default_flow_style=False)
        
    except Exception as e:
        logger.error(f"Memorize operation failed: {e}")
        error_result = {
            "status": "error",
            "message": str(e)
        }
        return yaml.dump(error_result, default_flow_style=False)


@mcp.tool()
def recall(query: str, domain_filter: Optional[List[str]] = None, 
           type_filter: Optional[List[str]] = None, limit: int = 5) -> str:
    """Hybrid search with RRF scoring and manifest intercept."""
    try:
        redis_client = get_redis_client()
        
        if query in ["*", "manifest"]:
            manifest = redis_client.get_manifest()
            if manifest:
                return yaml.dump(manifest, default_flow_style=False)
            else:
                return yaml.dump({"message": "No manifest found"}, default_flow_style=False)
        
        query_embedding = generate_embedding(query)
        
        results = redis_client.search_hybrid(
            query=query,
            query_embedding=query_embedding,
            domain_filter=domain_filter,
            type_filter=type_filter,
            limit=limit
        )
        
        formatted_results = {
            "query": query,
            "total_results": len(results),
            "results": []
        }
        
        for result in results:
            node_data = redis_client.get_node(result["id"])
            if node_data:
                formatted_result = {
                    "id": result["id"],
                    "domain": node_data.get("domain", ""),
                    "type": node_data.get("type", ""),
                    "content": node_data.get("content", ""),
                    "rrf_score": result.get("rrf_score", 0.0),
                    "text_rank": result.get("text_rank"),
                    "vector_rank": result.get("vector_rank"),
                    "links": result.get("links", {"inbound": [], "outbound": []}),
                    "created_at": node_data.get("created_at", 0.0)
                }
                formatted_results["results"].append(formatted_result)
        
        return yaml.dump(formatted_results, default_flow_style=False)
        
    except Exception as e:
        logger.error(f"Recall operation failed: {e}")
        error_result = {
            "status": "error",
            "message": str(e)
        }
        return yaml.dump(error_result, default_flow_style=False)


@mcp.tool()
def patch(node_id: str, operations: List[Dict[str, Any]]) -> str:
    """Apply atomic JSON patch operations to a node."""
    try:
        redis_client = get_redis_client()
        
        node_data = redis_client.get_node(node_id)
        if not node_data:
            result = {
                "status": "error",
                "message": f"Node {node_id} not found"
            }
            return yaml.dump(result, default_flow_style=False)
        
        success = redis_client.apply_patch(node_id, operations)
        
        if success:
            result = {
                "status": "success",
                "node_id": node_id,
                "operations_applied": len(operations),
                "updated_at": time.time()
            }
        else:
            result = {
                "status": "error",
                "message": "Failed to apply patch operations"
            }
        
        return yaml.dump(result, default_flow_style=False)
        
    except Exception as e:
        logger.error(f"Patch operation failed: {e}")
        error_result = {
            "status": "error",
            "message": str(e)
        }
        return yaml.dump(error_result, default_flow_style=False)


@mcp.tool()
def search_exact(query: str, limit: int = 10) -> str:
    """Pure BM25 text search for exact matches."""
    try:
        redis_client = get_redis_client()
        
        results = redis_client.search_exact(query, limit)
        
        formatted_results = {
            "query": query,
            "search_type": "exact_bm25",
            "total_results": len(results),
            "results": results
        }
        
        return yaml.dump(formatted_results, default_flow_style=False)
        
    except Exception as e:
        logger.error(f"Search exact operation failed: {e}")
        error_result = {
            "status": "error",
            "message": str(e)
        }
        return yaml.dump(error_result, default_flow_style=False)


@mcp.tool()
def inspect_node(node_id: str) -> str:
    """Retrieve raw node data with complete edge relationships."""
    try:
        redis_client = get_redis_client()
        
        node_data = redis_client.get_node(node_id)
        if not node_data:
            result = {
                "status": "error",
                "message": f"Node {node_id} not found"
            }
            return yaml.dump(result, default_flow_style=False)
        
        formatted_result = {
            "id": node_id,
            "domain": node_data.get("domain", ""),
            "type": node_data.get("type", ""),
            "content": node_data.get("content", ""),
            "created_at": node_data.get("created_at", 0.0),
            "metadata": node_data.get("metadata", {}),
            "links": node_data.get("links", {"inbound": [], "outbound": []}),
            "embedding_dimension": len(node_data.get("embedding", [])),
            "raw_data": node_data
        }
        
        return yaml.dump(formatted_result, default_flow_style=False)
        
    except Exception as e:
        logger.error(f"Inspect node operation failed: {e}")
        error_result = {
            "status": "error",
            "message": str(e)
        }
        return yaml.dump(error_result, default_flow_style=False)
