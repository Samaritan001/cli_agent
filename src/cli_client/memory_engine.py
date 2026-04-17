import math
import secrets
from collections import defaultdict
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import faiss
import numpy as np
from fastembed import TextEmbedding

from memory_llm import MemoryLLMBackend, NullMemoryLLM, build_memory_llm_from_env


def _stable_int_id() -> int:
    """Return a positive int64-friendly id for FAISS (avoid 128-bit uuid overflow)."""
    return secrets.randbits(63) or 1


@dataclass
class MemoryNode:
    id: int
    text: str
    fact_type: str  # 'world' or 'experience'
    timestamp: datetime
    embedding: np.ndarray
    entities: Dict[str, int]  # {entity_name: frequency}
    doc_length: int
    causes: Dict[int, int]  # {cause_node_id: causality_level}
    effects: Dict[int, int]  # {effect_node_id: causality_level}


class MemoryEngine:
    def __init__(self, llm: Optional[MemoryLLMBackend] = None):
        # CORE STORAGE:
        self.nodes: Dict[int, MemoryNode] = {}

        # --- LINK DATA STRUCTURES ---
        
        # 1. TEMPORAL LINKS: Sequential list of IDs
        # Stored as an ordered list where index reflects chronological arrival.
        # Provides O(1) adjacency lookups for time-based decay.
        self.temporal_stream: List[int] = []
        # 2. ENTITY LINKS: Inverted Index (Hash Map)
        # Structure: { "Entity_Name": [node_id1, node_id2] }
        # Allows O(1) retrieval of all facts related to a specific person or object.
        self.entity_index: Dict[str, List[int]] = {}
        self.entity_N = 5  # Max number of entities to extract from a context

        # 3. SEMANTIC LINKS: FAISS Index (HNSW)
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.semantic_index = faiss.IndexIDMap(faiss.IndexHNSWFlat(384, 32)) # Support custom ID, (Dimension, links per node)

        # 4. CAUSE LINKS: Adjacency List (Directed Graph)
        # Structure: { "Cause_Node_ID": ["Effect_Node_ID_1", "Effect_Node_ID_2"] }
        # Stored within MemoryNode
        self.cause_window = 3  # Number of recent nodes to consider for cause-effect linking

        # --- RECALL PARAMETERS ---
        # 1. Semantic Search Parameters
        self.semantic_K = 5  # Number of semantic neighbors to retrieve
        
        # 2. Entity Search Parameters
        self.k1 = 1.5  # BM25 parameter
        self.b = 0.75  # BM25 parameter
        self.avg_dl = 0  # Average document length for BM25 (updated dynamically)
        self.entity_K = 5 # Number of nodes to recall based on entity matching

        # 3. Cause-Effect Search Parameters
        self.cause_effect_K = 5  # Max number of cause/effect nodes to retrieve during recall

        # 4. Fusion Parameters
        self.k_rrf = 60  # Reciprocal Rank Fusion parameter for multi-channel fusion
        self.logical_weights = {"entity": 1, "semantic": 1, "cause": 1}  # Weights for multi-channel fusion

        # 5. TODO: Implement temporal recalling by absolute timestamp matching, also consider time decay functions
        self.temporal_decay_lambda = 0.1  # Lambda parameter for exponential decay of memory relevance over time

        # 7. Recent Memory and Threshold Parameters
        self.direct_temporal_window = 5  # Number of recent nodes directly added to recall results
        self.max_recall = 20  # Max number of nodes to return after recall fusion and reranking
        # self.min_recall_score = 0.5  # Minimum score threshold for a node to be included in recall results

        # 8. Token Limit for Recalled Memory Context
        self.memory_token_limit = 1000  # Max total tokens for recalled nodes (for LLM input)

        # 9. LLM Backend
        self.llm: MemoryLLMBackend = llm if llm is not None else NullMemoryLLM()

    @classmethod
    def from_env(cls) -> "MemoryEngine":
        """Construct engine with ``build_memory_llm_from_env()`` (feature flags + provider config)."""
        return cls(llm=build_memory_llm_from_env())

    def remember(self, context: str):
        """
        STAGE 1: REMEMBER
        Extracts facts and updates graph indices.
        """
        node_id = _stable_int_id()

        raw = list(self.embedding_model.embed([context]))
        embedding = np.asarray(raw, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        self.semantic_index.add_with_ids(embedding, np.array([node_id], dtype=np.int64))

        # TODO: Extract entities using LLM
        entities = self.extract_entities(context)
        for entity in entities:
            self.entity_index.setdefault(entity, []).append(node_id)
        context_tokens = context.lower().split()
        entity_counts = {entity: context_tokens.count(entity.lower()) for entity in entities}

        # TODO: Identify causes using LLM
        window_ids = self.temporal_stream[-self.cause_window :]
        causes = self.identify_causes(context, window_ids)
        for cause_id, level in causes.items():
            if cause_id in self.nodes:
                self.nodes[cause_id].effects[node_id] = level
        
        # Insert new memory node
        now = datetime.now(timezone.utc)
        new_node = MemoryNode(
            id=node_id,
            text=context,
            fact_type="experience",
            timestamp=now,
            embedding=embedding,
            doc_length=len(context_tokens),
            entities=entity_counts,
            causes=causes,
            effects={},
        )

        # Store node and update Temporal Stream
        self.nodes[node_id] = new_node
        self.temporal_stream.append(node_id)
        n = len(self.nodes)
        self.avg_dl = (self.avg_dl * (n - 1) + len(context_tokens)) / max(n, 1)

        return node_id

    def recall(self, query: str) -> str:
        """
        STAGE 2: RECALL
        Multi-channel retrieval: Semantic, Entity, and Graph Traversal.
        """
        # 1. Semantic Search (O(log n) with HNSW)
        if not self.nodes:
            return ""

        raw_q = list(self.embedding_model.embed([query]))
        query_embedding = np.asarray(raw_q, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, ids = self.semantic_index.search(query_embedding, self.semantic_K)
        semantic_candidates = [int(i) for i in ids[0].tolist() if int(i) >= 0]

        # 2. Entity Lookup (O(1) via Hash Map)
        entity_candidates = self.entity_bm25(query)

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
        )[: self.cause_effect_K]

        # 4. Fusion: merge semantic, entity, and cause lists via Reciprocal Rank Fusion (RRF).
        # TODO: match a certain time range
        rrf_results = self.rrf(
            {"semantic": semantic_candidates, "entity": entity_candidates, "cause": cause_effect_candidates}
        )
        if not rrf_results:
            return ""

        # 5. Temporal recency: decay fused scores by age (hours since node timestamp).
        scored_candidates = self.temporal_boost(rrf_results)
        # 6. Neural reranking: reorder candidates (LLM backend; identity if using NullMemoryLLM).
        ranked_candidates = self.neural_rerank(list(scored_candidates.keys()), query)

        # 7. Final ordering: prepend recent stream, then fill remainder from reranked list (deduped later).
        w = self.direct_temporal_window
        rank_budget = max(0, self.max_recall - w)
        # TODO: Prune using score threshold
        final_ids = self.temporal_stream[-w:] + ranked_candidates[:rank_budget]

        # 8. Render context string: walk ids in order, skip duplicates, stop at token budget.
        memory_context = ""
        token_count = 0
        examined_ids: Set[int] = set()
        for nid in final_ids:
            if nid in examined_ids:
                continue
            examined_ids.add(nid)
            node = self.nodes.get(nid)
            if node is None:
                continue
            if token_count + node.doc_length > self.memory_token_limit:
                break
            token_count += node.doc_length
            memory_context += f"{node.text}\n{'-' * 10}\n"

        return memory_context

    def reflect(self, node_ids: List[int]) -> str:
        """
        STAGE 3: REFLECT
        Synthesizes raw nodes into an "Observation" or "Mental Model".
        In a real system, this sends the nodes to an LLM to resolve contradictions.
        """
        # TODO: design a better consolidation algorithm or AI handling
        # TODO: prune out-dated or low-confidence memory nodes
        # Store this as a new 'observation' fact type node
        facts = [self.nodes[nid].text for nid in node_ids if nid in self.nodes]
        summary = self.llm.reflect_synthesize(facts)

        obs_id = self.remember(summary)
        self.nodes[obs_id].fact_type = "observation"
        return summary
    
    # TODO: Function: Convert history to context string
    # TODO: Function: Convert files, images, and other formats to text

    # TODO: Function (extractEntities): Extract entities from context using LLM
    def extract_entities(self, context: str) -> List[str]:
        return self.llm.extract_entities(context)

    # TODO: Function (identifyCauses): Identify cause-effect relationships using LLM
    def identify_causes(self, context: str, memory_window: List[int]) -> Dict[int, int]:
        # Placeholder for cause relationship identification logic using LLM
        # TODO: Need to define the level of causality (e.g. 0-5) and ask LLM to identify
        return self.llm.identify_causes(context, memory_window)

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
                numerator = f_q * (self.k1 + 1)
                denominator = f_q + self.k1 * (1 - self.b + self.b * (L_d / avg_dl))
                scores[node_id] += idf * (numerator / denominator)

        # STEP 3: Rank and return top-k
        sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return sorted_ids[: self.entity_K]

    def rrf(self, ranked_lists: Dict[str, List[int]]) -> Dict[int, float]:
        """
        Reciprocal Rank Fusion (RRF) to merge multiple retrieval channels.
        """
        rrf_scores: Dict[int, float] = defaultdict(float)
        for category, r_list in ranked_lists.items():
            for rank, node_id in enumerate(r_list):
                score = 1.0 / (self.k_rrf + rank + 1)
                rrf_scores[node_id] += score * self.logical_weights.get(category, 1)
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
            recency_boost = float(np.exp(-self.temporal_decay_lambda * hours_old))
            
            # Apply Recency Boost to RRF Score
            final_results[node_id] = rrf_score * recency_boost

        return final_results
    
    # TODO: Neural Reranking using LLM
    def neural_rerank(self, candidate_ids: List[int], query: str) -> List[int]:
        return self.llm.neural_rerank(candidate_ids, query)


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
