from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum
from vllm import AsyncLLMEngine
from vllm.sequence import Logprob


class StepStatus(Enum):
    DIRECT_GENERATED = 0
    DIRECT_SUCCESS = 1
    DIRECT_FAILED = 2
    RAG_GENERATED = 3
    RAG_FINISHED = 4
    RAG_FAILED = 5

@dataclass
class Step:
    status: StepStatus
    content: str
    search_query: Optional[str] = None
    best_docid: Optional[str] = None
    score: Optional[float] = None
    logprobs: Optional[List[Dict[int, Logprob]]] = None # to build query

@dataclass
class UncertaintyScore:
    logprobs: List[Tuple[int, float]]
    perplexity: float
    ln_entropy: float
    energy_score: float
    eigen_score: float

@dataclass
class LLMOutputWithUncertainty:
    greedy_response: str
    sample_responses: List[str]
    uncertainty: UncertaintyScore