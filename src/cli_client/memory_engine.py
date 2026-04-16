import uuid
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np
from fastembed import TextEmbedding
import faiss
from typing import List, Dict, Set, Any

@dataclass
class MemoryNode:
    id: int
    text: str
    fact_type: str  # 'world' or 'experience'
    timestamp: datetime
    embedding: np.ndarray
    entities: Dict[str, int] # {entity_name: frequency}
    doc_length: int
    causes: Dict[int, int] # {cause_node_id: causality_level}
    effects: Dict[int, int] # {effect_node_id: causality_level}

class MemoryEngine:
    def __init__(self):
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


    def remember(self, context: str):
        """
        STAGE 1: REMEMBER
        Extracts facts and updates graph indices.
        """
        
        node_id = int(uuid.uuid4())
        
        embedding = np.array([self.embedding_model.embed([context])]).astype('float32')
        self.semantic_index.add_with_ids(embedding, np.array([node_id]))

        # TODO: Extract entities using LLM
        entities = self.extract_entities(context)
        for entity in entities:
            if entity not in self.entity_index:
                self.entity_index[entity] = []
            self.entity_index[entity].append(node_id)
        context_tokens = context.lower().split()
        entity_counts = {entity: context_tokens.count(entity.lower()) for entity in entities}

        # TODO: Identify cause-effect relationships using LLM
        causes = self.identify_causes(context, self.temporal_stream[-self.cause_window:])
        for cause, level in causes.items():
            if cause in self.nodes:
                self.nodes[cause].effects[node_id] = level

        # Insert new memory node
        new_node = MemoryNode(
            id=node_id,
            text=context,
            fact_type="experience",
            timestamp=datetime.now(),
            embedding=embedding,
            doc_length=len(context_tokens),
            entities=entity_counts,
            causes=causes,
            effects={}
        )

        # Store node and update Temporal Stream
        self.nodes[node_id] = new_node
        self.temporal_stream.append(node_id)
        self.avg_dl = (self.avg_dl * len(self.nodes) + len(context_tokens)) / (len(self.nodes) + 1)

        return node_id

    def recall(self, query: str) -> str:
        """
        STAGE 2: RECALL
        Multi-channel retrieval: Semantic, Entity, and Graph Traversal.
        """
        # 1. Semantic Search (O(log n) with HNSW)
        query_embedding = self.embedding_model.embed([query]).astype('float32')
        distances, ids = self.semantic_index.search(query_embedding, self.semantic_K)
        semantic_candidates = ids[0].tolist()
        # candidate_ids.update({str(id): float(distance) for id, distance in zip(ids[0].tolist(), distances[0].tolist())})

        # 2. Entity Lookup (O(1) via Hash Map)
        entity_candidates = self.entity_bm25(query)

        # 3. Cause-Effect Spreading (Graph Traversal)
        cause_effect_candidates = {}
        seed_ids = set(semantic_candidates) | set(entity_candidates)
        for cid in seed_ids:
            # update considers overlapping nodes
            cause_effect_candidates.update(self.nodes[cid].causes)
            cause_effect_candidates.update(self.nodes[cid].effects)
        cause_effect_candidates = sorted(cause_effect_candidates, key=cause_effect_candidates.get, reverse=True)[:self.cause_effect_K]

        # 4. Fusion of semantic, entity, and causal nodes using Reciprocal Rank Fusion (RRF)
        rrf_results = self.rrf({"semantic": semantic_candidates, "entity": entity_candidates, "cause": cause_effect_candidates})

        # 5. Temporal Recency (Simple slice of the stream)
        # TODO: match a certain time range
        scored_candidates = self.temporal_boost(rrf_results)
        
        # 6. TODO: Neuro Reranking with LLM
        ranked_cadidates = self.neural_rerank(list(scored_candidates.keys()), query)

        # 7. Add recent memory nodes directly
        # TODO: Prune using score threshold
        final_ids = self.temporal_stream[-self.direct_temporal_window:] + ranked_cadidates[:self.max_recall - self.direct_temporal_window]

        # 8. Convert recalled nodes to a context string for LLM input
        memory_context = ""
        token_count = 0
        examined_ids = set()  # To avoid duplicates
        for node_id in final_ids:
            if node_id in examined_ids:
                continue
            examined_ids.add(node_id)
            node = self.nodes[node_id]
            # Token-limit assessment and pruning
            if token_count + node.doc_length > self.memory_token_limit:
                break
            token_count += node.doc_length
            memory_context += f"{node.text}\n{'-'*10}\n"

        return memory_context


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
        obs_id = self.remember(summary)
        self.nodes[obs_id].fact_type = "observation"
        return summary
    
    # TODO: Function: Convert history to context string
    # TODO: Function: Convert files, images, and other formats to text

    # TODO: Function (extractEntities): Extract entities from context using LLM
    def extract_entities(self, context: str) -> List[str]:
        # Placeholder for entity extraction logic using LLM
        current_entities = list(self.entity_index.keys())
        return []

    # TODO: Function (identifyCauses): Identify cause-effect relationships using LLM
    def identify_causes(self, context: str, memory_window: List[int]) -> Dict[int, int]:
        # Placeholder for cause relationship identification logic using LLM
        # TODO: Need to define the level of causality (e.g. 0-5) and ask LLM to identify
        return {} # {cause_node_id: causality_level}
    
    def entity_bm25(self, query: str) -> List[str]:
        query_entities = self.extract_entities(query)

        scores: Dict[str, float] = {}
        N = len(self.nodes)

        for entity in query_entities:
            # 1. Calculate IDF for this token
            n_q = len(self.entity_index.get(entity, []))
            if n_q == 0: continue
            idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
            
            # 2. Score each candidate containing this token
            for node_id in self.entity_index[entity]:
                node = self.nodes[node_id]
                f_q = node.entities.get(entity, 0) # Simple term frequency
                
                # BM25 Formula components
                L_d = node.doc_length
                numerator = f_q * (self.k1 + 1)
                denominator = f_q + self.k1 * (1 - self.b + self.b * (L_d / self.avg_dl))
                
                scores[node_id] += idf * (numerator / denominator)

        # STEP 3: Rank and return top-k
        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        return sorted_ids[:self.entity_K]

    def rrf(self, ranked_lists: Dict[str, List[str]]) -> Dict[int, float]:
        """
        Implements Reciprocal Rank Fusion (RRF) to merge multiple retrieval channels.
        """
        rrf_scores = {}
        for category, r_list in ranked_lists.items():
            for rank, node_id in enumerate(r_list):
                # RRF Formula: 1 / (k + rank)
                score = 1.0 / (self.k_rrf + rank + 1)
                rrf_scores[node_id] += score * self.logical_weights.get(category, 1) # Apply logical weight for this category
        return rrf_scores

    def temporal_boost(self, rrf_results: Dict[str, float]) -> Dict[int, float]:
        """
        Applies recency and temporal boosts to the fused RRF results.
        """
        final_results = {}
        now = datetime.now(timezone.utc)

        for node_id, rrf_score in rrf_results.items():
            # Recency Boost: Exponential decay based on hours since creation
            hours_old = (now - node.timestamp).total_seconds() / 3600
            recency_boost = np.exp(-self.temporal_decay_lambda * hours_old) # Simulated decay
            
            # Final combined weight
            combined_weight = rrf_score * recency_boost
            
            final_results[node_id] = combined_weight
            
        return final_results
    
    # TODO: Neural Reranking using LLM
    def neural_rerank(self, candidate_ids: List[int], query: str) -> List[int]:
        # Placeholder for neural reranking logic using LLM
        return candidate_ids



# Example Usage
engine = MemoryEngine()

# Remembering facts
id1 = engine.remember("User is a student at UCLA", entities=["User", "UCLA"])
id2 = engine.remember("User received a parking ticket in Westwood", entities=["Westwood"], cause_of=id1)

# Recalling facts
memories = engine.recall("Tell me about the user's location", target_entities=["UCLA"])
for m in memories:
    print(f"Recalled: {m.id} : {m.timestamp} : {m.text} ({m.fact_type})")