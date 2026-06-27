
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np

@dataclass
class MemoryNode:
    id: int
    text: str
    fact_type: str  # "EXPERIENCE", "OBSERVATION"
    timestamp: datetime
    embedding: np.ndarray
    entities: Dict[str, int]  # {entity_name: frequency}
    doc_length: int
    causes: Dict[int, int]  # {cause_node_id: causality_level}
    effects: Dict[int, int]  # {effect_node_id: causality_level}
    message_id_range: List[int] = field(default_factory=lambda: [0, 0])  # Tracks start and end IDs of the memory

@dataclass
class MemoryConfig:
    memory_dir: str = "./memory_docs"
    
    # LINK DATA STRUCTURE PARAMETERS
    entity_N: int = 5 # 2. Max number of entities to extract from a context
    faiss_dim: int = 384 # 3. Dimension of text embeddings
    faiss_links_per_node: int = 32 # 3. Number of links per node
    cause_window: int = 3 # 4. Number of recent nodes to consider for cause-effect linking
    
    # RECALL PARAMETERS
    # 1.Semantic Search Parameters
    semantic_K: int = 5 # Number of semantic neighbors to retrieve
    # 2. Entity Search Parameters
    k1: float = 1.5 # BM25 k1
    b: float = 0.75 # BM25 b
    entity_K: int = 5 # Number of nodes to recall based on entity matching
    # 3. Cause-Effect Search Parameters
    cause_effect_K: int = 5 # Max number of cause/effect nodes to retrieve during recall
    # 4. Fusion Parameters
    k_rrf: int = 60 # Reciprocal Rank Fusion parameter for multi-channel fusion
    logical_weights: Dict[str, float] = field(default_factory=lambda: {"entity": 1, "semantic": 1, "cause": 1}) # 4. Weights for multi-channel fusion
    # 5. Temporal Decay Parameters
    temporal_decay_lambda: float = 0.1 # Lambda parameter for exponential decay of memory relevance over time
    # 6. Minimum score threshold for a node to be included in recall results
    min_recall_score: float = 0.5 # Minimum score threshold for a node to be included in recall results
    # 7. Threshold Parameters
    max_recall: int = 20 # Max number of nodes to return after recall fusion and reranking
    # 8. Token Limit for Recalled Memory Context
    memory_token_limit: int = 2000 # Max total tokens for recalled nodes (for LLM input)
    
    # SUMMARY PARAMETERS
    # Generation Parameters
    summary_token_limit: int = 500 # Max tokens for generated summary
