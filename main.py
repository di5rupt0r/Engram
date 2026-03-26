"""Main entry point for the Engram MCP server."""

import logging
import os
from engram.server import memorize, recall, patch as patch_node, search_exact, inspect_node

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('ENGRAM_LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the MCP server."""
    logger.info("Starting Engram MCP Server...")
    
    try:
        # Test Redis connection
        from engram.redis.client import EngramRedisClient
        redis_client = EngramRedisClient()
        logger.info("Redis connection established")
        
        # Test embedding generation
        from engram.embeddings.provider import generate_embedding, get_embedding_dimension
        test_embedding = generate_embedding("Test connection")
        logger.info(f"Embedding generation working - dimension: {get_embedding_dimension()}")
        
        # Test index setup
        from engram.index.setup import setup_redis_index
        if setup_redis_index():
            logger.info("Redis index setup successful")
        else:
            logger.warning("Redis index setup failed - continuing anyway")
        
        logger.info("Engram MCP Server ready and operational")
        
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise


if __name__ == "__main__":
    main()
