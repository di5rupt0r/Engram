"""Main entry point for the Engram MCP server with CORS middleware."""

import logging
import os
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from engram.server import mcp

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('ENGRAM_LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the MCP server with CORS middleware."""
    logger.info("Starting Engram MCP Server with CORS...")
    
    try:
        # Initialize Redis connection
        from engram.server import get_redis_client
        redis_client = get_redis_client()
        logger.info("Redis connection established")
        
        # Get the underlying ASGI app from FastMCP
        app = mcp._app
        
        # Wrap with CORS middleware
        app = CORSMiddleware(
            app=app,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Start uvicorn with CORS-enabled app
        host = os.getenv('MCP_HOST', '0.0.0.0')
        port = int(os.getenv('MCP_PORT', '8000'))
        
        logger.info(f"Starting SSE server with CORS on {host}:{port}")
        uvicorn.run(app, host=host, port=port, access_log=False)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


if __name__ == "__main__":
    main()
