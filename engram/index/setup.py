"""Redis index setup for Engram knowledge graph."""

import redis
import logging
from typing import Optional

logger = logging.getLogger(__name__)

INDEX_NAME = "engram_nodes"
PREFIX = "node:"
VECTOR_DIM = 384
VECTOR_DISTANCE = "COSINE"
VECTOR_ALGORITHM = "FLAT"


def create_index(redis_client: redis.Redis, drop_existing: bool = False) -> bool:
    """Create the RediSearch index for Engram nodes using raw commands."""
    try:
        if drop_existing:
            try:
                redis_client.execute_command("FT.DROPINDEX", INDEX_NAME)
                logger.info(f"Dropped existing index: {INDEX_NAME}")
            except Exception:
                pass
        
        # Define the schema for RediSearch on JSON
        # Vector dimension is 384 for all-MiniLM-L6-v2
        schema = [
            "FT.CREATE", INDEX_NAME,
            "ON", "JSON",
            "PREFIX", "1", PREFIX,
            "SCHEMA",
            "$.domain", "AS", "domain", "TAG",
            "$.type", "AS", "type", "TAG",
            "$.content", "AS", "content", "TEXT", "SORTABLE",
            "$.created_at", "AS", "created_at", "NUMERIC", "SORTABLE",
            "$.embedding", "AS", "embedding", "VECTOR", "FLAT", "6",
            "TYPE", "FLOAT32",
            "DIM", str(VECTOR_DIM),
            "DISTANCE_METRIC", VECTOR_DISTANCE
        ]
        
        redis_client.execute_command(*schema)
        logger.info(f"Successfully created index: {INDEX_NAME}")
        return True
        
    except Exception as e:
        if "Index already exists" in str(e):
            logger.info(f"Index {INDEX_NAME} already exists")
            return True
        logger.error(f"Failed to create index: {e}")
        return False


def setup_redis_index(host: str = "localhost", port: int = 6379, password: Optional[str] = None) -> bool:
    """Complete Redis index setup process."""
    try:
        redis_client = redis.Redis(
            host=host, port=port, password=password, decode_responses=True
        )
        redis_client.ping()
        
        # Ensure we can run FT.CREATE (idempotency handled inside create_index)
        return create_index(redis_client, drop_existing=False)
        
    except Exception as e:
        logger.error(f"Redis setup failed: {e}")
        return False
