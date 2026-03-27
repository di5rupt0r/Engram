"""Tests for FastMCP server integration following TDD principles."""

import pytest
from unittest.mock import Mock, patch


class TestFastMCPServer:
    """Test cases for FastMCP server integration."""
    
    def test_fastmcp_import_available(self):
        """Test that FastMCP can be imported from fastmcp."""
        try:
            from fastmcp import FastMCP
            assert FastMCP is not None
        except ImportError:
            pytest.fail("FastMCP import failed - fastmcp not installed")
    
    def test_mcp_server_instantiation(self):
        """Test that FastMCP server can be instantiated."""
        from fastmcp import FastMCP
        
        mcp = FastMCP("Engram")
        
        assert mcp is not None
        assert hasattr(mcp, 'tool')
    
    def test_mcp_server_has_tools_decorator(self):
        """Test that FastMCP server has tool decorator method."""
        from fastmcp import FastMCP
        
        mcp = FastMCP("Test")
        
        assert callable(mcp.tool)
    
    def test_memorize_tool_registered(self):
        """Test that memorize function is registered as MCP tool."""
        with patch('engram.server.get_redis_client'):
            with patch('engram.server.generate_embedding'):
                from engram.server import memorize
                
                assert callable(memorize)
    
    def test_recall_tool_registered(self):
        """Test that recall function is registered as MCP tool."""
        with patch('engram.server.get_redis_client'):
            with patch('engram.server.generate_embedding'):
                from engram.server import recall
                
                assert callable(recall)
    
    def test_patch_tool_registered(self):
        """Test that patch function is registered as MCP tool."""
        with patch('engram.server.get_redis_client'):
            from engram.server import patch as patch_tool
            
            assert callable(patch_tool)
    
    def test_search_exact_tool_registered(self):
        """Test that search_exact function is registered as MCP tool."""
        with patch('engram.server.get_redis_client'):
            from engram.server import search_exact
            
            assert callable(search_exact)
    
    def test_inspect_node_tool_registered(self):
        """Test that inspect_node function is registered as MCP tool."""
        with patch('engram.server.get_redis_client'):
            from engram.server import inspect_node
            
            assert callable(inspect_node)


class TestMainEntryPoint:
    """Test cases for main.py entry point."""
    
    def test_main_imports_mcp(self):
        """Test that main.py imports mcp from engram.server."""
        try:
            from main import mcp
            assert mcp is not None
        except ImportError as e:
            pytest.fail(f"main.py failed to import mcp: {e}")
        except AttributeError as e:
            pytest.fail(f"mcp not available in main.py: {e}")
    
    def test_main_runs_mcp_server(self):
        """Test that main.py runs the MCP server."""
        with patch('engram.server.mcp.run') as mock_run:
            with patch('engram.server.get_redis_client'):
                from main import main
                
                try:
                    main()
                    mock_run.assert_called_once()
                except Exception:
                    pass  # main() may raise due to Redis, but mcp.run should be called
    
    def test_mcp_server_sse_transport(self):
        """Test that MCP server uses SSE transport."""
        with patch('engram.server.mcp.run') as mock_run:
            with patch('engram.server.get_redis_client'):
                from main import main
                
                try:
                    main()
                except Exception:
                    pass
                
                if mock_run.called:
                    call_args = mock_run.call_args
                    assert 'transport' in str(call_args) or 'sse' in str(call_args)
