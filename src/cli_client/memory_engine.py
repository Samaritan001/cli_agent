import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Set, Any

@dataclass
class MemoryNode:
    id: str
    text: str
    fact_type: str  # 'world' or 'experience'
    timestamp: datetime
    embedding: List[float] = None  # Simulated vector for Semantic Search

class MemoryEngine:
    def __init__(self):
        # CORE STORAGE: The "Epistemic Substrate"
        self.nodes: Dict[str, MemoryNode] = {}

        # --- LINK DATA STRUCTURES ---
        
        # 1. TEMPORAL LINKS: Sequential list of IDs
        # Stored as an ordered list where index reflects chronological arrival.
        # Provides O(1) adjacency lookups for time-based decay.
        self.temporal_stream: List[str] = []

        # 2. ENTITY LINKS: Inverted Index (Hash Map)
        # Structure: { "Entity_Name": [node_id1, node_id2] }
        # Allows O(1) retrieval of all facts related to a specific person or object.
        self.entity_index: Dict[str, List[str]] = {}

        # 3. SEMANTIC LINKS: Simulated Vector Index
        # In production, this is an HNSW graph or Vector DB.
        # Here, it stores mappings for cosine similarity comparisons.
        self.semantic_map: Dict[str, List[float]] = {}

        # 4. CAUSAL LINKS: Adjacency List (Directed Graph)
        # Structure: { "Cause_Node_ID": ["Effect_Node_ID_1", "Effect_Node_ID_2"] }
        # Explicitly maps cause-and-effect as identified by the LLM.
        self.causal_graph: Dict[str, List[str]] = {}

    def retain(self, conversation_text: str, entities: List[str] = None, cause_of: str = None):
        """
        STAGE 1: RETAIN
        Extracts facts and updates graph indices.
        """
        # TODO: add embedding to node
        node_id = str(uuid.uuid4())
        new_node = MemoryNode(
            id=node_id,
            text=conversation_text,
            fact_type="experience",
            timestamp=datetime.now()
        )
        
        # Store node and update Temporal Stream
        self.nodes[node_id] = new_node
        self.temporal_stream.append(node_id)

        # Update Entity Index
        if entities:
            for entity in entities:
                if entity not in self.entity_index:
                    self.entity_index[entity] = []
                self.entity_index[entity].append(node_id)

        # Update Causal Graph
        if cause_of and cause_of in self.nodes:
            if cause_of not in self.causal_graph:
                self.causal_graph[cause_of] = []
            self.causal_graph[cause_of].append(node_id)

        return node_id

    def recall(self, query: str, target_entities: List[str] = None) -> List[MemoryNode]:
        """
        STAGE 2: RECALL
        Multi-channel retrieval: Semantic, Entity, and Graph Traversal.
        """
        candidate_ids: Set[str] = set()

        # TODO: Semantic Search based on embeddings
        # 1. Entity Lookup (O(1) via Hash Map)
        if target_entities:
            for entity in target_entities:
                candidate_ids.update(self.entity_index.get(entity, []))

        # 2. Bounded Spreading Activation (Graph Traversal)
        # If we found a 'cause', also recall the 'effect'.
        refined_candidates = list(candidate_ids)
        for cid in refined_candidates:
            if cid in self.causal_graph:
                candidate_ids.update(self.causal_graph[cid])

        # 3. Temporal Recency (Simple slice of the stream)
        # TODO: match a certain time range
        candidate_ids.update(self.temporal_stream[-5:])

        # TODO: Fusion
        # TODO: Neuro Reranking with LLM
        # TODO: Token-limit assessment and pruning

        return [self.nodes[nid] for nid in candidate_ids]

    def reflect(self, node_ids: List[str]) -> str:
        """
        STAGE 3: REFLECT
        Synthesizes raw nodes into an "Observation" or "Mental Model".
        In a real system, this sends the nodes to an LLM to resolve contradictions.
        """
        facts = [self.nodes[nid].text for nid in node_ids]
        # TODO: design a better consolidation algorithm or AI handling
        # TODO: prune out-dated or low-confidence memory nodes
        summary = f"Synthesized Knowledge: {'; '.join(facts)}"
        
        # Store this as a new 'observation' fact type node
        obs_id = self.retain(summary)
        self.nodes[obs_id].fact_type = "observation"
        return summary

# TODO: Implement AI or Algorithmic memory content and metadata pre-processing, recall query processing and neuro reranking, and reflect consolidation


# Example Usage
engine = MemoryEngine()

# Remembering facts
id1 = engine.retain("User is a student at UCLA", entities=["User", "UCLA"])
id2 = engine.retain("User received a parking ticket in Westwood", entities=["Westwood"], cause_of=id1)

# Recalling facts
memories = engine.recall("Tell me about the user's location", target_entities=["UCLA"])
for m in memories:
    print(f"Recalled: {m.id} : {m.timestamp} : {m.text} ({m.fact_type})")