from typing import Dict, List, Set, Optional
import json
import logging

LOG_FORMAT = "\033[32m%(levelname)s\033[0m:    %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("manual_generator")

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
            logging.info(f"Registering tool summary '{name}'")
            self._summaries[name] = summary

    def register_tool(self, name: str, manual: str):
        """Adds a tool to the master registry (not yet visible to the LLM)."""
        logging.info(f"Registering tool manual '{name}'")
        self._registry[name] = manual
    
    def check_tool(self, name: str) -> bool:
        return name in self._registry

    def inject_tool(self, name: str) -> bool:
        """Fetch from registry and move into the active context."""
        if name in self._registry:
            self._active_tools.add(name)
            logging.info(f"Injecting tool manual '{name}'")
            return True
        return False

    def prune_tool(self, name: str):
        """Remove a specific tool from the context after a specific call."""
        if name in self._active_tools:
            self._active_tools.remove(name)
            logging.info(f"Pruning tool manual '{name}'")
    
    def flush_loadout(self):
        """Clear all tool descriptions to reset context to a 'zero-pollution' state."""
        self._active_tools.clear()
        logging.info("Flushing all tool manuals from context")

    def get_all_summaries(self) -> List[dict]:
        """Returns all summaries of the tools."""
        return '\n\n'.join([f"Tool Name: {name}\nSummary: {summary}" for name, summary in self._summaries.items()])

    def get_all_manuals(self) -> List[dict]:
        """Returns the 'tools' parameter for the LLM API call."""
        delimeter = f"\n\n{'*'*30}\n\n"
        return delimeter.join([f"Tool Name: {name}\nManual: {self._registry[name]}" for name in self._active_tools])

class AgentContextManager:
    def __init__(self, system_instruction: Optional[str] = "You are a helpful assistant that can call tools to get information or perform actions."):
        self.system_instruction = system_instruction
        self.history = []  # List of message dicts: {"role": "...", "content": "..."}
        self.tool_manager = ToolManualManager()
    
    def add_message(self, role, content):
        """Adds a standard message to the history."""
        self.history.append({"role": role, "content": content})

    def register_summary(self, summaries: Dict[str, str]):
        self.tool_manager.register_summary(summaries)

    def register_tool(self, name: str, manual: str):
        self.tool_manager.register_tool(name, manual)
    
    def check_tool(self, name: str) -> bool:
        return self.tool_manager.check_tool(name)

    def inject_tool(self, name: str) -> bool:
        return self.tool_manager.inject_tool(name)
    
    def prune_tool(self, name: str):
        self.tool_manager.prune_tool(name)
    
    def get_last_message(self):
        if len(self.history) > 0:
            return "role: " + self.history[-1]["role"] + "\ncontent:\n" + self.history[-1]["content"]
        return ""

    def get_full_context(self):
        """
        Builds the message list for the LLM. 
        Structure: [System] + [History] + [Dynamic Tool Message]
        """
        messages = []
        
        # 1. Base System Instruction
        messages.append({"role": "system", "content": self.system_instruction})
        
        # 2. Past Conversation History
        messages.extend(self.history)
        
        # 3. Tool Summaries
        summaries = self.tool_manager.get_all_summaries()
        if summaries == "":
            messages.append({"role": "system", "content": "No tools are currently available."})
        else:
            messages.append({"role": "system", "content": f"Summaries of all available tools:\n{summaries}"})
            # 4. Dynamic Tool Injection (only active tools)
            manuals = self.tool_manager.get_all_manuals()
            if manuals != "":
                messages.append({"role": "system", "content": f"Full API manuals of active tools:\n{manuals}"})
        
        return messages

        



