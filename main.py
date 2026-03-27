"""Main entry point for the Engram MCP server."""

import logging
import os
from engram.server import mcp

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
        # Initialize Redis connection and components
        from engram.server import get_redis_client
        redis_client = get_redis_client()
        logger.info("Redis connection established")
        
        # Start FastMCP SSE server
        host = os.getenv('MCP_HOST', '0.0.0.0')
        port = int(os.getenv('MCP_PORT', '8000'))
        
        logger.info(f"Starting SSE server on {host}:{port}")
        mcp.run(transport='sse', host=host, port=port)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


if __name__ == "__main__":
    main()
