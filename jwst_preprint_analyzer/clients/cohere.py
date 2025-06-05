"""Cohere API client for reranking."""

import logging
from typing import List, Dict, Union, Optional

import cohere

logger = logging.getLogger(__name__)


class CohereClient:
    """Wrapper for Cohere reranking API."""
    
    def __init__(self, api_key: Optional[str], model: str = 'rerank-v3.5'):
        self.client = None
        self.model = model
        
        if not api_key:
            logger.warning("COHERE_API_KEY not found. Reranking will be skipped (using original order).")
        else:
            try:
                self.client = cohere.ClientV2(api_key)
                available_cohere_models = self.client.models.list()
                available_models = [m.name for m in available_cohere_models.models]
                if model not in available_models and not model.startswith('rerank-'):
                    logger.warning(f"Cohere reranker model '{model}' not found in available models. Check model name.")
            except Exception as e:
                logger.error(f"An unexpected error occurred during Cohere client initialization: {e}")
                self.client = None
    
    def rerank_snippets(self, query: str, snippets: List[str], 
                       top_k: int) -> List[Dict[str, Union[str, Optional[float]]]]:
        """Rerank snippets using Cohere API based on the query."""
        if not self.client or not snippets:
            return [{'snippet': s, 'score': None} for s in snippets[:top_k]]

        logger.debug(f"Reranking {len(snippets)} snippets with Cohere model '{self.model}' for query: '{query}'")
        try:
            non_empty_snippets = [s for s in snippets if s and s.strip()]
            if not non_empty_snippets:
                logger.warning("All extracted snippets were empty after stripping.")
                return []

            results = self.client.rerank(
                query=query,
                documents=non_empty_snippets,
                top_n=top_k,
                model=self.model
            )

            reranked_data = [
                {'snippet': non_empty_snippets[result.index], 'score': result.relevance_score} 
                for result in results.results
            ]

            logger.info(f"Reranked scores (top 3): {[f'{r.relevance_score:.3f}' for r in results.results[:3]]}")
            return reranked_data

        except Exception as e:
            logger.error(f"Unexpected error during Cohere reranking: {e}")
            return [{'snippet': s, 'score': None} for s in non_empty_snippets[:top_k]]