from typing import Dict, List, Set, Optional
import json
import logging

LOG_FORMAT = "\033[32m%(levelname)s\033[0m:    %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("managers")


class ToolManualManager:
    def __init__(self):
        # summary of all tools
        self._summaries: Dict[str, str] = {}

        # registry of the tool manuals, keyed by server name
        self._registry: Dict[str, str] = {}
        
        # The "Active Loadout"
        # These are the tools currently injected into the LLM's prompt/API call.
        self._active_tools: Set[str] = set()

    def register_summary(self, summaries: Dict[str, str]):
        """Adds a tool summary to the master summary registry."""
        for name, summary in summaries.items():
            logger.info(f"Registering tool summary '{name}'")
            self._summaries[name] = summary

    def register_tool(self, name: str, manual: str):
        """Adds a tool to the master registry (not yet visible to the LLM)."""
        logger.info(f"Registering tool manual '{name}'")
        self._registry[name] = manual
    
    def check_tool(self, name: str) -> bool:
        return name in self._registry

    def inject_tool(self, name: str) -> bool:
        """Fetch from registry and move into the active context."""
        if name in self._registry:
            self._active_tools.add(name)
            logger.info(f"Injecting tool manual '{name}'")
            return True
        return False

    def prune_tool(self, name: str):
        """Remove a specific tool from the context after a specific call."""
        if name in self._active_tools:
            self._active_tools.remove(name)
            logger.info(f"Pruning tool manual '{name}'")
    
    def flush_loadout(self):
        """Clear all tool descriptions to reset context to a 'zero-pollution' state."""
        self._active_tools.clear()
        logger.info("Flushing all tool manuals from context")

    def get_all_summaries(self) -> str:
        """Returns all summaries of the tools."""
        return '\n\n'.join([f"Tool Name: {name}\nSummary: {summary}" for name, summary in self._summaries.items()])

    def get_all_manuals(self) -> str:
        """Returns the 'tools' parameter for the LLM API call."""
        delimeter = f"\n\n{'*'*30}\n\n"
        return delimeter.join([f"Tool Name: {name}\nManual: {self._registry[name]}" for name in self._active_tools])
    
    def get_tool_info(self) -> Dict[str, str]:
        """
        Return tool summaries and active tool manuals
        """

        tool_info = {
            "tool_summaries": self.get_all_summaries(),
            "tool_manuals": self.get_all_manuals()
        }

        return tool_info




