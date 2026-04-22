import math
import asyncio
import secrets
from collections import defaultdict
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Set
import glob
import json
import os

from memory_config import MemoryNode, MemoryConfig

import logging

LOG_FORMAT = "\033[32m%(levelname)s\033[0m:    %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("memory_engine")

import faiss
import numpy as np
from fastembed import TextEmbedding

# from memory_llm import MemoryLLMBackend, NullMemoryLLM, build_memory_llm_from_env
from model import llm_side_request, llm_side_request_async, LanguageModelConfig

DEFAULT_SESSION_MEMORY_TEMPLATE = """
# Session Title
_A short and distinctive 5-10 word descriptive title for the session. Super info dense, no filler_

# Current State
_What is actively being worked on right now? Pending tasks not yet completed. Immediate next steps._

# Task specification
_What did the user ask to build? Any design decisions or other explanatory context_

# Files and Functions
_What are the important files? In short, what do they contain and why are they relevant?_

# Workflow
_What bash commands are usually run and in what order? How to interpret their output if not obvious?_

# Errors & Corrections
_Errors encountered and how they were fixed. What did the user correct? What approaches failed and should not be tried again?_

# Codebase and System Documentation
_What are the important system components? How do they work/fit together?_

# Learnings
_What has worked well? What has not? What to avoid? Do not duplicate items from other sections_

# Key results
_If the user asked a specific output such as an answer to a question, a table, or other document, repeat the exact result here_

# Worklog
_Step by step, what was attempted, done? Very terse summary for each step_
"""

def _stable_int_id() -> int:
    """Return a positive int64-friendly id for FAISS (avoid 128-bit uuid overflow)."""
    return secrets.randbits(63) or 1

def _strip_json_fence(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines)
    return s.strip()


class MemoryEngine:
    def __init__(self, config: Optional[MemoryConfig] = None, model_config: Optional[LanguageModelConfig] = None):
        # CORE STORAGE
        self.nodes: Dict[int, MemoryNode] = {}
        self.config = config or MemoryConfig()
        self.model_config = model_config or LanguageModelConfig()
        self._lock = asyncio.Lock()
        
        # --- LINK DATA STRUCTURES ---

        # 1. TEMPORAL LINKS: Sequential list of IDs
        # Stored as an ordered list where index reflects chronological arrival.
        # Provides O(1) adjacency lookups for time-based decay.
        self.temporal_stream: List[int] = []
        
        # 2. ENTITY LINKS: Inverted Index (Hash Map)
        # Structure: { "Entity_Name": [node_id1, node_id2] }
        # Allows O(1) retrieval of all facts related to a specific person or object.
        self.entity_index: Dict[str, List[int]] = {}

        # 3. SEMANTIC LINKS: FAISS Index (HNSW)
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.semantic_index = faiss.IndexIDMap(faiss.IndexHNSWFlat(self.config.faiss_dim, self.config.faiss_links_per_node)) # Support custom ID, (Dimension, links per node)

        # 4. CAUSE LINKS: Adjacency List (Directed Graph)
        # Structure: { "Cause_Node_ID": ["Effect_Node_ID_1", "Effect_Node_ID_2"] }
        # Stored within MemoryNode

        # --- RECALL PARAMETERS ---
        # 0. Recall Range
        self.latest_message_id = 0
        # 2. Entity Search Parameters
        self.avg_dl = 0.0 # Average document length for BM25
        # 5. TODO: Implement temporal recalling by absolute timestamp matching, also consider time decay functions
        # 7. TODO: Minimum score threshold for a node to be included in recall results
        

    @staticmethod
    def _safe_json_loads(raw_content: str) -> Optional[Dict]:
        try:
            parsed = json.loads(_strip_json_fence(raw_content))
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _validate_entities_payload(payload: Optional[Dict], max_entities: int) -> List[str]:
        if not payload or "entities" not in payload or not isinstance(payload["entities"], list):
            return []
        entities: List[str] = []
        for item in payload["entities"]:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned and cleaned not in entities:
                    entities.append(cleaned)
            if len(entities) >= max_entities:
                break
        return entities
    
    @staticmethod
    def _validate_causalities_payload(payload: Optional[Dict], allowed_ids: Set[int]) -> Dict[int, int]:
        if not payload or "causalities" not in payload or not isinstance(payload["causalities"], list):
            return {}
        causalities: Dict[int, int] = {}
        for item in payload["causalities"]:
            if not isinstance(item, dict):
                continue
            memory_id = item.get("memory_id")
            level = item.get("level")
            if not isinstance(memory_id, int) or not isinstance(level, int):
                continue
            if memory_id not in allowed_ids:
                continue
            causalities[memory_id] = max(0, min(level, 3))
        return causalities

    @staticmethod
    def _validate_ranks_payload(payload: Optional[Dict], candidate_ids: Set[int]) -> List[int]:
        if not payload or "ranks" not in payload or not isinstance(payload["ranks"], list):
            return []
        ranked: List[tuple[int, int]] = []
        seen_ids: Set[int] = set()
        seen_ranks: Set[int] = set()
        for item in payload["ranks"]:
            if not isinstance(item, dict):
                continue
            memory_id = item.get("memory_id")
            rank = item.get("rank")
            if not isinstance(memory_id, int) or not isinstance(rank, int):
                continue
            if memory_id not in candidate_ids or memory_id in seen_ids or rank <= 0 or rank in seen_ranks:
                continue
            seen_ids.add(memory_id)
            seen_ranks.add(rank)
            ranked.append((rank, memory_id))
        ranked.sort(key=lambda x: x[0])
        return [memory_id for _, memory_id in ranked]

    # TODO: better summarize triggering algorithm
    def should_summarize_turn(self, user_input: str, assistant_output: str, had_tool_calls: bool, pending_turns: int) -> bool:
        importance = 0.0
        total_chars = len(user_input.strip()) + len(assistant_output.strip())
        if total_chars >= self.config.summary_min_chars:
            importance += 1.0
        if had_tool_calls:
            importance += 1.0
        if "?" in user_input:
            importance += 0.5
        if pending_turns >= self.config.summary_force_after_turns:
            return True
        return importance >= self.config.summary_importance_threshold

    # TODO: Original Claude Code summary prompt requires direct write to file ability.
    # TODO: Convert files, images, and other formats to text
    @staticmethod
    def summarize_turn_buffer(history) -> str:
        context = self._summary_system_prompt()
        result = llm_side_request(context, config=self.model_config, history=history)
        summary = self._safe_json_loads(result.get("content", ""))
        if summary.stripe():
            return summary
        logger.warning("memory_llm summarize_turn schema validation failed: %s", result.get("content", ""))
        return ""

    @staticmethod
    def _summary_system_prompt() -> str:
        return f"""IMPORTANT: This message and these instructions are NOT part of the actual user conversation. Do NOT include any references to "note-taking", "session notes extraction", or these update instructions in the notes content.

Based on the user conversation above (EXCLUDING this note-taking instruction message as well as system prompt, or any past session summaries), update the session memory.

The current content structure is:
<current_notes_content>
{DEFAULT_SESSION_MEMORY_TEMPLATE}
</current_notes_content>

Your ONLY task is to use the Edit tool to update the notes file, then stop. You can make multiple edits (update every section as needed) - make all Edit tool calls in parallel in a single message. Do not call any other tools.

CRITICAL RULES FOR EDITING:
- The file must maintain its exact structure with all sections, headers, and italic descriptions intact
-- NEVER modify, delete, or add section headers (the lines starting with '#' like # Task specification)
-- NEVER modify or delete the italic _section description_ lines (these are the lines in italics immediately following each header - they start and end with underscores)
-- The italic _section descriptions_ are TEMPLATE INSTRUCTIONS that must be preserved exactly as-is - they guide what content belongs in each section
-- ONLY update the actual content that appears BELOW the italic _section descriptions_ within each existing section
-- Do NOT add any new sections, summaries, or information outside the existing structure
- Do NOT reference this note-taking process or instructions anywhere in the notes
- It's OK to skip updating a section if there are no substantial new insights to add. Do not add filler content like "No info yet", just leave sections blank/unedited if appropriate.
- Write DETAILED, INFO-DENSE content for each section - include specifics like file paths, function names, error messages, exact commands, technical details, etc.
- For "Key results", include the complete, exact output the user requested (e.g., full table, full answer, etc.)
- Do not include information that's already in the CLAUDE.md files included in the context
- Keep each section under ~{self.config.summary_token_limit} tokens/words - if a section is approaching this limit, condense it by cycling out less important details while preserving the most critical information
- Focus on actionable, specific information that would help someone understand or recreate the work discussed in the conversation

STRUCTURE PRESERVATION REMINDER:
Each section has TWO parts that must be preserved exactly as they appear in the current file:
1. The section header (line starting with #)
2. The italic description line (the _italicized text_ immediately after the header - this is a template instruction)

You ONLY update the actual content that comes AFTER these two preserved lines. The italic description lines starting and ending with underscores are part of the template structure, NOT content to be edited or removed.

REMEMBER: Use the Edit tool in parallel and stop. Do not continue after the edits. Only include insights from the actual user conversation, never from these note-taking instructions. Do not delete or change section headers or italic _section descriptions_.
"""

    def load_memory(self):
        if not os.path.exists(self.config.memory_dir):
            logger.error(f"Directory {self.config.memory_dir} does not exist.")
            return

        json_files = glob.glob(os.path.join(self.config.memory_dir, "*.json"))
        loaded_nodes = []

        for file_path in json_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 1. Reconstruct complex types
            # Convert ISO string back to datetime
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            
            # Convert list back to numpy array
            data['embedding'] = np.array(data['embedding'], dtype='float32')
            
            # JSON keys are strings; convert causes/effects keys back to int
            data['causes'] = {int(k): v for k, v in data['causes'].items()}
            data['effects'] = {int(k): v for k, v in data['effects'].items()}
            
            # Create the dataclass instance
            node = MemoryNode(**data)
            loaded_nodes.append(node)

        if not loaded_nodes:
            return

        # 2. Update CORE STORAGE
        self.nodes = {node.id: node for node in loaded_nodes}

        # 3. Update TEMPORAL LINKS
        # Sort by timestamp to ensure chronological order in the stream
        loaded_nodes.sort(key=lambda x: x.timestamp)
        self.temporal_stream = [node.id for node in loaded_nodes]

        # 4. Update ENTITY LINKS (Inverted Index)
        self.entity_index = {}
        total_doc_length = 0
        
        for node in loaded_nodes:
            total_doc_length += node.doc_length
            for entity_name in node.entities.keys():
                if entity_name not in self.entity_index:
                    self.entity_index[entity_name] = []
                self.entity_index[entity_name].append(node.id)

        # 5. Update BM25 Parameters
        self.avg_dl = total_doc_length / len(loaded_nodes)

        # 6. Update SEMANTIC LINKS (FAISS Index)
        # We clear the index first to avoid duplicates if this is called multiple times
        self.semantic_index = faiss.IndexIDMap(faiss.IndexHNSWFlat(self.config.faiss_dim, self.config.faiss_links_per_node))
        
        embeddings = np.array([node.embedding for node in loaded_nodes]).astype('float32')
        ids = np.array([node.id for node in loaded_nodes]).astype('int64')
        
        self.semantic_index.add_with_ids(embeddings, ids)

        if loaded_nodes:
            self.latest_message_id = max((node.message_id_range[1] for node in loaded_nodes), default=0)

        logger.info(f"Successfully hydrated memory: {len(self.nodes)} nodes loaded and indexed.")

    async def aload_memory(self):
        async with self._lock:
            await asyncio.to_thread(self.load_memory)

    def save_memory(self):
        if not os.path.exists(self.config.memory_dir):
            logger.info(f"Directory {self.config.memory_dir} does not exist. Creating it.")
            os.makedirs(self.config.memory_dir)

        cnt = 0
        for id in self.temporal_stream[::-1]:
            # Define file path using the node id
            file_path = os.path.join(self.config.memory_dir, f"{id}.json")
            if os.path.exists(file_path):
                break  # Stop if we encounter an existing file, assuming all previous nodes are already saved

            # Convert dataclass to dictionary
            node_data = asdict(self.nodes[id])

            # Handle non-serializable fields
            # 1. Convert datetime to ISO format string
            if isinstance(node_data['timestamp'], datetime):
                node_data['timestamp'] = node_data['timestamp'].isoformat()

            # 2. Convert numpy embedding to a list
            if isinstance(node_data['embedding'], np.ndarray):
                node_data['embedding'] = node_data['embedding'].tolist()

            # Write to JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(node_data, f, indent=4)
            
            cnt += 1

        logger.info(f"Successfully saved {cnt} nodes to '{self.config.memory_dir}/'")

    async def asave_memory(self):
        async with self._lock:
            await asyncio.to_thread(self.save_memory)
   
    def remember(self, context: str, message_id_range: Optional[List[int]] = None) -> int:
        """
        STAGE 1: REMEMBER
        Extracts facts and updates graph indices.
        """
        node_id = _stable_int_id()
        if message_id_range is None:
            message_id_range = [0, 0]

        raw = list(self.embedding_model.embed([context]))
        embedding = np.asarray(raw, dtype=np.float32)
        if embedding.ndim == 1:
            embedding2D = embedding.reshape(1, -1)
        else:
            embedding2D = embedding

        self.semantic_index.add_with_ids(embedding2D, np.array([node_id], dtype=np.int64))

        # Extract entities using LLM
        entities = self.extract_entities(context)
        for entity in entities:
            self.entity_index.setdefault(entity, []).append(node_id)
        context_tokens = context.lower().split()
        entity_counts = {entity: context_tokens.count(entity.lower()) for entity in entities}

        # Identify causes using LLM
        window_ids = self.temporal_stream[-self.config.cause_window :]
        causes = self.identify_causes(context, window_ids)
        for cause_id, level in causes.items():
            if cause_id in self.nodes:
                self.nodes[cause_id].effects[node_id] = level
        
        # Insert new memory node
        now = datetime.now(timezone.utc)
        new_node = MemoryNode(
            id=node_id,
            text=context,
            fact_type="EXPERIENCE",
            timestamp=now,
            embedding=embedding,
            doc_length=len(context_tokens),
            entities=entity_counts,
            causes=causes,
            effects={},
            message_id_range=message_id_range
        )

        # Store node and update Temporal Stream
        self.nodes[node_id] = new_node
        self.temporal_stream.append(node_id)
        n = len(self.nodes)
        self.avg_dl = (self.avg_dl * (n - 1) + len(context_tokens)) / max(n, 1)
        self.latest_message_id = max(self.latest_message_id, message_id_range[1])

        return node_id

    async def aremember(self, context: str, message_id_range: List[int]):
        async with self._lock:
            return await asyncio.to_thread(self.remember, context, message_id_range)

    def recall(self, query: str, earliest_history_id: Optional[int] = 0) -> Tuple[int, List[MemoryNode]]:
        """
        STAGE 2: RECALL
        Multi-channel retrieval: Semantic, Entity, and Graph Traversal.
        """
        def is_valid(nid: int) -> bool:
            if earliest_history_id == 0: return True
            node = self.nodes.get(nid)
            # Exclude node if it overlaps with preserved history
            return node is not None and node.message_id_range[1] < earliest_history_id

        # 1. Semantic Search (O(log n) with HNSW)
        if not self.nodes:
            return []

        raw_q = list(self.embedding_model.embed([query]))
        query_embedding = np.asarray(raw_q, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, ids = self.semantic_index.search(query_embedding, self.config.semantic_K)
        semantic_candidates = [int(i) for i in ids[0].tolist() if int(i) >= 0]
        semantic_candidates = list(dict.fromkeys(semantic_candidates))  # Deduplicate while preserving order

        # 2. Entity Lookup (O(1) via Hash Map)
        entity_candidates = self.entity_bm25(query)
        entity_candidates = list(dict.fromkeys(entity_candidates))  # Deduplicate while preserving order

        # 3. Cause-Effect Spreading (Graph Traversal)
        merged_cause_effect: Dict[int, int] = {}
        seed_ids: Set[int] = set(semantic_candidates) | set(entity_candidates)
        for cid in seed_ids:
            # update considers overlapping nodes
            node = self.nodes.get(cid)
            if node is None:
                continue
            for nbr_id, lvl in node.causes.items():
                merged_cause_effect[nbr_id] = max(merged_cause_effect.get(nbr_id, 0), lvl)
            for nbr_id, lvl in node.effects.items():
                merged_cause_effect[nbr_id] = max(merged_cause_effect.get(nbr_id, 0), lvl)

        # Top-K cause/effect neighbors by merged causal strength (feeds the "cause" RRF list).
        cause_effect_candidates = sorted(
            merged_cause_effect.keys(),
            key=lambda x: merged_cause_effect[x],
            reverse=True,
        )[: self.config.cause_effect_K]
        cause_effect_candidates = list(dict.fromkeys(cause_effect_candidates))  # Deduplicate while preserving order

        # Filter candidates based on earliest_history_id
        semantic_candidates = [nid for nid in semantic_candidates if is_valid(nid)]
        entity_candidates = [nid for nid in entity_candidates if is_valid(nid)]
        cause_effect_candidates = [nid for nid in cause_effect_candidates if is_valid(nid)]

        # 4. Fusion: merge semantic, entity, and cause lists via Reciprocal Rank Fusion (RRF).
        # TODO: match a certain time range
        rrf_results = self.rrf(
            {"semantic": semantic_candidates, "entity": entity_candidates, "cause": cause_effect_candidates}
        )
        if not rrf_results:
            return []

        # 5. Temporal recency: decay fused scores by age (hours since node timestamp).
        scored_candidates = self.temporal_boost(rrf_results)
        # 6. Neural reranking: reorder candidates (LLM backend; identity if using NullMemoryLLM).
        ranked_candidates = self.neural_rerank(list(scored_candidates.keys()), query)

        # 7. Final ordering: prepend recent stream, then fill remainder from reranked list (deduped later).
        # TODO: Prune using score threshold
        final_ids = ranked_candidates[:self.config.max_recall]
        
        # 8. Compute token budget
        final_nodes = []
        token_count = 0
        examined_ids: Set[int] = set()
        for nid in final_ids:
            if nid in examined_ids:
                continue
            examined_ids.add(nid)
            node = self.nodes.get(nid)
            if node is None:
                continue
            if token_count + node.doc_length > self.config.memory_token_limit:
                break
            token_count += node.doc_length
            final_nodes.append(node)

        # return the memory nodes
        return self.latest_message_id, final_nodes

    async def arecall(self, query: str, earliest_history_id: Optional[int] = 0) -> Tuple[int, List[MemoryNode]]:
        async with self._lock:
            return await asyncio.to_thread(self.recall, query, earliest_history_id)

    # TODO: Function: Implement reflection using LLM
    def reflect(self, node_ids: List[int]) -> str:
        """
        STAGE 3: REFLECT
        Synthesizes raw nodes into an "Observation" or "Mental Model".
        In a real system, this sends the nodes to an LLM to resolve contradictions.
        """
        # TODO: design a better consolidation algorithm or AI handling
        # TODO: prune out-dated or low-confidence memory nodes
        # Store this as a new 'observation' fact type node
        # facts = [self.nodes[nid].text for nid in node_ids if nid in self.nodes]
        # summary = self.llm.reflect_synthesize(facts)

        # obs_id = self.remember(summary)
        # self.nodes[obs_id].fact_type = "OBSERVATION"
        # return summary
        pass
 
    
    # Extract entities from context using LLM
    def extract_entities(self, context: str) -> List[str]:
        # result = self.llm.extract_entities(context, self.config.entity_N)
        system_prompt = self._entities_system_prompt(self.config.entity_N)
        result = llm_side_request(context, system_prompt, config=self.model_config)
        parsed = self._safe_json_loads(result.get("content", ""))
        entities = self._validate_entities_payload(parsed, self.config.entity_N)
        if entities:
            return entities
        logger.warning("memory_llm extract_entities schema validation failed: %s", result.get("content", ""))
        return []

    async def aextract_entities(self, context: str) -> List[str]:
        system_prompt = self._entities_system_prompt(self.config.entity_N)
        result = await llm_side_request_async(context, system_prompt, config=self.model_config)
        parsed = self._safe_json_loads(result.get("content", ""))
        entities = self._validate_entities_payload(parsed, self.config.entity_N)
        if entities:
            return entities
        logger.warning("memory_llm extract_entities schema validation failed: %s", result.get("content", ""))
        return []

    
    @staticmethod
    def _entities_system_prompt(max_entities: int) -> str:
        return (
            "You extract named entities for memory indexing. "
            "Return ONLY valid JSON with this exact shape: "
            '{"entities": ["Entity1", "Entity2"]}. '
            f"Include at most {max_entities} entities. "
            "Prefer people, organizations, locations, and key proper nouns; "
            "use short surface forms as they appear in the text; "
            "no duplicate meanings; use an empty array if there are none."
        )

    # Identify cause-effect relationships using LLM
    def identify_causes(self, current_memory: str, memory_window: List[int]) -> Dict[int, int]:
        past_memories = [{"memory_id": self.nodes[id].id, "memory_content": self.nodes[id].text} for id in memory_window]
        memories = json.dumps(past_memories)
        context = f"Past Memories:\n{memories}\n\nCurrent Memory:\n{current_memory}"
        
        # result = self.llm.identify_causes(context, len(memory_window))
        
        system_prompt = self._causes_system_prompt(len(memory_window))
        result = llm_side_request(context, system_prompt, config=self.model_config)
        parsed = self._safe_json_loads(result.get("content", ""))
        causes = self._validate_causalities_payload(parsed, set(memory_window))
        if causes:
            return causes
        logger.warning("memory_llm identify_causes schema validation failed: %s", result.get("content", ""))
        return {}

    async def aidentify_causes(self, current_memory: str, memory_window: List[int]) -> Dict[int, int]:
        past_memories = [{"memory_id": self.nodes[id].id, "memory_content": self.nodes[id].text} for id in memory_window]
        memories = json.dumps(past_memories)
        context = f"Past Memories:\n{memories}\n\nCurrent Memory:\n{current_memory}"
        system_prompt = self._causes_system_prompt(len(memory_window))
        result = await llm_side_request_async(context, system_prompt, config=self.model_config)
        parsed = self._safe_json_loads(result.get("content", ""))
        causes = self._validate_causalities_payload(parsed, set(memory_window))
        if causes:
            return causes
        logger.warning("memory_llm identify_causes schema validation failed: %s", result.get("content", ""))
        return {}

    @staticmethod
    def _causes_system_prompt(max_causes: int) -> str:
        return (
            f"You are a Causal Logic Engine. Your task is to analyze the causal relationship "
            f"between {max_causes} previous 'Source Memories' and one 'Current Memory'.\n\n"
            "### CAUSALITY SCALE:\n"
            "0: NO RELATION - The memories are independent or share only surface-level topics/entities.\n"
            "1: WEAK/INDIRECT - The Source provides helpful background context but is not necessary for the Current memory.\n"
            "2: STRONG/DIRECT - The Source is a clear precursor or contributor to the events in the Current memory.\n"
            "3: CRITICAL/NECESSARY - The Current memory would not exist or cannot be understood without the Source.\n\n"
            "### CONSTRAINTS:\n"
            "- Ignore 'Entity Matching': Do not assign a level > 0 just because both memories mention the same person or place.\n"
            "- Focus on 'Logical Flow': Does the Source memory explain *why* or *how* the Current memory occurred?\n"
            f"- Output exactly {max_causes} entries in the JSON array.\n\n"
            "### OUTPUT FORMAT:\n"
            "Return ONLY valid JSON in this shape:\n"
            '{"causalities": [{"memory_id": integer, "level": integer}]}'
        )    


    def entity_bm25(self, query: str) -> List[int]:
        query_entities = self.extract_entities(query)

        scores: Dict[int, float] = defaultdict(float)
        N = len(self.nodes)
        if N == 0:
            return []

        avg_dl = max(self.avg_dl, 1e-6)

        for entity in query_entities:
            # STEP 1: Calculate IDF for this token
            n_q = len(self.entity_index.get(entity, []))
            if n_q == 0:
                continue
            idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
            
            # STEP 2. Score each candidate containing this token
            for node_id in self.entity_index[entity]:
                node = self.nodes[node_id]
                f_q = node.entities.get(entity, 0)
                L_d = node.doc_length
                numerator = f_q * (self.config.k1 + 1)
                denominator = f_q + self.config.k1 * (1 - self.config.b + self.config.b * (L_d / avg_dl))
                scores[node_id] += idf * (numerator / denominator)

        # STEP 3: Rank and return top-k
        sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return sorted_ids[:self.config.entity_K]

    def rrf(self, ranked_lists: Dict[str, List[int]]) -> Dict[int, float]:
        """
        Reciprocal Rank Fusion (RRF) to merge multiple retrieval channels.
        """
        rrf_scores: Dict[int, float] = defaultdict(float)
        for category, r_list in ranked_lists.items():
            for rank, node_id in enumerate(r_list):
                score = 1.0 / (self.config.k_rrf + rank + 1)
                rrf_scores[node_id] += score * self.config.logical_weights.get(category, 1)
        return dict(rrf_scores)

    def temporal_boost(self, rrf_results: Dict[int, float]) -> Dict[int, float]:
        """
        Applies recency and temporal boosts to the fused RRF results.
        """
        final_results: Dict[int, float] = {}
        now = datetime.now(timezone.utc)

        for node_id, rrf_score in rrf_results.items():
            node = self.nodes.get(node_id)
            if node is None:
                continue
            ts = node.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            # Recency Boost: exponential decay based on hours since creation
            hours_old = (now - ts).total_seconds() / 3600.0
            recency_boost = float(np.exp(-self.config.temporal_decay_lambda * hours_old))
            
            # Apply Recency Boost to RRF Score
            final_results[node_id] = rrf_score * recency_boost

        return final_results
    
    # Neural Reranking using LLM
    def neural_rerank(self, candidate_ids: List[int], query: str) -> List[int]:
        if not candidate_ids:
            return []
        past_memories = [{"memory_id": self.nodes[id].id, "memory_content": self.nodes[id].text} for id in candidate_ids]
        memories = json.dumps(past_memories)
        context = f"Past Memories:\n{memories}\n\nUser Query:\n{query}"
        
        # result = self.llm.neural_rerank(context, len(memory_window))
        
        system_prompt = self._rerank_system_prompt(len(candidate_ids))
        result = llm_side_request(context, system_prompt, config=self.model_config)
        parsed = self._safe_json_loads(result.get("content", ""))
        ranked_ids = self._validate_ranks_payload(parsed, set(candidate_ids))
        if ranked_ids:
            return ranked_ids
        logger.warning("memory_llm neural_rerank schema validation failed: %s", result.get("content", ""))
        return candidate_ids

    async def aneural_rerank(self, candidate_ids: List[int], query: str) -> List[int]:
        if not candidate_ids:
            return []
        past_memories = [{"memory_id": self.nodes[id].id, "memory_content": self.nodes[id].text} for id in candidate_ids]
        memories = json.dumps(past_memories)
        context = f"Past Memories:\n{memories}\n\nUser Query:\n{query}"
        system_prompt = self._rerank_system_prompt(len(candidate_ids))
        result = await llm_side_request_async(context, system_prompt, config=self.model_config)
        parsed = self._safe_json_loads(result.get("content", ""))
        ranked_ids = self._validate_ranks_payload(parsed, set(candidate_ids))
        if ranked_ids:
            return ranked_ids
        logger.warning("memory_llm neural_rerank schema validation failed: %s", result.get("content", ""))
        return candidate_ids

    @staticmethod
    def _rerank_system_prompt(num_memories: int) -> str:
        return (
            "You are a Semantic Relevance Auditor. Your task is to rank a set of memory nodes "
            f"based on their utility in answering the User's next query.\n\n"
            "### RANKING CRITERIA:\n"
            "1. DIRECT ANSWER: Does the memory contain the specific information requested?\n"
            "2. CONTEXTUAL SUPPORT: Does the memory provide necessary background or 'why' for the query?\n"
            "3. TEMPORAL RELEVANCE: If the query implies a sequence, is this memory a logical part of that timeline?\n"
            "4. NOISE REDUCTION: If a memory is unrelated or only shares generic keywords, rank it lowest.\n\n"
            "### CONSTRAINTS:\n"
            f"- You must rank exactly {num_memories} memory nodes.\n"
            f"- Assign a unique integer 'rank' from 1 to {num_memories}, where 1 is the MOST relevant and "
            f"{num_memories} is the LEAST relevant.\n"
            "- Do not allow ties; every memory must have a distinct rank.\n\n"
            "### OUTPUT FORMAT:\n"
            "Return ONLY valid JSON with this exact structure:\n"
            '{"ranks": [{"memory_id": integer, "rank": integer}]}'
        )



# Example usage (module import does not run side effects)
if __name__ == "__main__":
    engine = MemoryEngine()
    id1 = engine.remember("User is a student at UCLA")
    id2 = engine.remember("User received a parking ticket in Westwood")
    out = engine.recall("Tell me about the user's location")
    print(out)
    for nid in (id1, id2):
        n = engine.nodes[nid]
        print(f"{n.id}: {n.timestamp} : {n.text} ({n.fact_type})")
