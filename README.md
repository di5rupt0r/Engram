# Engram: Agentic Knowledge Graph

A production-ready MCP (Model Context Protocol) server implementing an Agentic Knowledge Graph as a Single Source of Truth (SSoT) for multiple autonomous AI agents.

## Architecture

- **Database**: Redis with RedisJSON and RediSearch modules
- **Vector Embeddings**: FastEmbed (CPU-optimized, 384-dimensional)
- **MCP Server**: FastMCP
- **Search**: Hybrid BM25 + Vector with Reciprocal Rank Fusion (RRF)
- **Testing**: Test-Driven Development (TDD) with pytest

## Features

### MCP Tools

1. **memorize(domain, type, content, metadata=None)**
   - Store nodes with automatic embedding generation
   - Updates system manifest for omni-awareness
   - Returns YAML-formatted success status

2. **recall(query, domain_filter=None, type_filter=None, limit=5)**
   - Hybrid search with RRF scoring
   - Manifest intercept for "*" or "manifest" queries
   - Returns YAML-formatted search results

3. **patch(node_id, operations)**
   - Apply atomic JSON patch operations
   - Supports set, delete, append operations
   - Returns YAML-formatted operation status

4. **search_exact(query, limit=10)**
   - Pure BM25 text search
   - Optimized for exact proper noun matching
   - Returns YAML-formatted results

5. **inspect_node(node_id)**
   - Raw node retrieval with edge relationships
   - Complete metadata and embedding info
   - Returns YAML-formatted node data

## Installation

```bash
# Clone repository
git clone <repository-url>
cd Engram

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Run tests
python -m pytest tests/ -v
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# Engram Configuration
ENGRAM_LOG_LEVEL=INFO
ENGRAM_MAX_RESULTS=50
ENGRAM_RRF_K=60

# FastEmbed Configuration
ENGRAM_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENGRAM_CACHE_DIR=/tmp/fastembed_cache
```

## Usage

```bash
# Run the MCP server
python main.py

# Run with custom configuration
ENGRAM_LOG_LEVEL=DEBUG python main.py
```

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_embeddings.py -v

# Run with coverage
python -m pytest tests/ --cov=engram --cov-report=html
```

### Code Quality

```bash
# Format code
black engram/ tests/

# Lint code
ruff check engram/ tests/

# Type checking
mypy engram/
```

## Architecture Decisions

- **TDD Approach**: All functionality developed with failing tests first
- **Single Responsibility**: Each function under 20 lines
- **No Mocking**: Real Redis operations (tests use proper mocking)
- **Environment Variables**: All configuration via environment
- **Error Exposure**: Verbose error reporting for debugging
- **Atomic Commits**: Each feature is a single commit

## Requirements

- Python 3.11+
- Redis 7.0+ with RedisJSON and RediSearch modules
- Local Redis instance on localhost:6379 (configurable)

## Performance

- **Embedding Generation**: CPU-optimized with FastEmbed
- **Vector Dimensions**: 384 (balance of quality and speed)
- **RRF Scoring**: k=60 for optimal ranking fusion
- **Index Schema**: Optimized for hybrid search patterns

## License

MIT License - see LICENSE file for details.
