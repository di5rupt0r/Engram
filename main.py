import logging
import os
import uvicorn
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import Response
from engram.server import mcp
from pythonjsonlogger import jsonlogger

# Configure JSON logging
log_level = getattr(logging, os.getenv('ENGRAM_LOG_LEVEL', 'INFO'))
logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(log_level)

# Create a local logger for main
main_logger = logging.getLogger(__name__)


def main():
    """Main entry point for the MCP server with CORS and JSON logging."""
    main_logger.info("Starting Engram MCP Server with CORS and JSON logging...")
    
    try:
        # Initialize Redis connection
        from engram.server import get_redis_client
        get_redis_client()
        main_logger.info("Redis connection established")
        
        # Get the underlying ASGI app from FastMCP
        app = mcp._app
        
        # Wrap with CORS middleware
        # Explicitly setting allow_methods to include OPTIONS
        app = CORSMiddleware(
            app=app,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS", "PATCH", "DELETE"],
            allow_headers=["*"],
            expose_headers=["*"],
        )
        
        # Add a custom exception handler or middleware if needed to catch 405s 
        # But CORSMiddleware should handle OPTIONS. 
        # If it doesn't, we can add a simple middleware to intercept /sse OPTIONS.
        
        @app.middleware("http")
        async def handle_options_preflight(request, call_next):
            if request.method == "OPTIONS":
                return Response(status_code=204, headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS, PATCH, DELETE",
                    "Access-Control-Allow-Headers": "*"
                })
            return await call_next(request)
        
        # Start uvicorn
        host = os.getenv('MCP_HOST', '0.0.0.0')
        port = int(os.getenv('MCP_PORT', '8000'))
        
        main_logger.info(f"Starting SSE server on {host}:{port}")
        uvicorn.run(app, host=host, port=port, access_log=False)
        
    except Exception as e:
        main_logger.error(f"Failed to start server: {e}")
        raise


if __name__ == "__main__":
    main()
