import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from cli_client.managers import ToolManualManager

def test_tool_manual_manager():
    manager = ToolManualManager()
    
    # Test Registration
    manager.register_summary({"math": "Does math", "weather": "Gets weather"})
    assert "math" in manager._summaries
    
    manager.register_tool("math", "Math Manual Content")
    assert manager.check_tool("math") is True
    assert manager.check_tool("weather") is False

    # Test Injection
    assert manager.inject_tool("math") is True
    assert "math" in manager._active_tools
    assert manager.inject_tool("unknown") is False

    # Test Info Retrieval
    info = manager.get_tool_info("google")
    assert "Does math" in info["tool_summaries"]
    assert "Math Manual Content" in info["tool_manuals"]

    # Test Pruning & Flushing
    manager.prune_tool("math")
    assert "math" not in manager._active_tools
    
    manager.inject_tool("math")
    manager.flush_loadout()
    assert len(manager._active_tools) == 0

if __name__ == "__main__":
    test_tool_manual_manager()
    print("✅ All tests passed!")