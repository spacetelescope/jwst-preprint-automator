"""JWST science content analysis."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..models import JWSTScienceLabelerModel
from ..clients.openai import OpenAIClient
from ..clients.cohere import CohereClient
from ..clients.gpt_reranker import GPTReranker
from ..processing.text_extractor import TextExtractor

logger = logging.getLogger(__name__)


class ScienceAnalyzer:
    """Analyzes papers for JWST science content."""
    
    SCIENCE_KEYWORDS = [
        "jwst", "james webb space telescope", "webb telescope", "webb",
        "nircam", "nirspec", "miri", "niriss", "fgs",
        "program id", "proposal id",
    ]
    SCIENCE_KEYWORDS_LOWER = sorted(
        [k.lower() for k in SCIENCE_KEYWORDS], 
        key=len, reverse=True
    )
    
    def __init__(self, 
                 openai_client: OpenAIClient,
                 cohere_client: CohereClient,
                 text_extractor: TextExtractor,
                 prompts: Dict[str, str],
                 top_k_snippets: int = 15,
                 reranker_threshold: float = 0.1,
                 validate_llm: bool = False,
                 gpt_reranker: Optional[GPTReranker] = None):
        self.openai_client = openai_client
        self.cohere_client = cohere_client
        self.gpt_reranker = gpt_reranker
        self.text_extractor = text_extractor
        self.prompts = prompts
        self.top_k_snippets = top_k_snippets
        self.reranker_threshold = reranker_threshold
        self.validate_llm = validate_llm
        
    def analyze(self, arxiv_id: str, text_path: Path) -> Dict:
        """Analyze paper for JWST science content using extracted snippets."""
        logger.info(f"Analyzing JWST science content for {arxiv_id}")
        
        if not text_path.exists():
            logger.warning(f"Text file not found for {arxiv_id}, cannot analyze science.")
            return {"jwstscience": -1.0, "reason": "Analysis failed: Text file missing", 
                   "quotes": [], "error": "missing_text_file"}

        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                paper_text = f.read()
        except Exception as e:
            logger.error(f"Failed to read text file {text_path}: {e}")
            return {"jwstscience": -1.0, "reason": "Analysis failed: Cannot read text file", 
                   "quotes": [], "error": "read_error"}

        # Extract snippets
        all_snippets = self.text_extractor.extract_relevant_snippets(
            paper_text, self.SCIENCE_KEYWORDS_LOWER
        )

        if not all_snippets:
            logger.info(f"No relevant keywords found for science analysis in {arxiv_id}.")
            return {"jwstscience": 0.0, "quotes": [], 
                   "reason": "No relevant keywords (e.g., JWST, instruments) found in text."}

        # Rerank snippets
        rerank_query = self.prompts.get('rerank_science_query')
        if not rerank_query:
            logger.error("Rerank science query prompt ('rerank_science_query.txt') not found or empty.")
            return {"jwstscience": -1.0, "reason": "Analysis failed: Missing rerank science query prompt", 
                   "quotes": [], "error": "prompt_missing"}

        # Use GPT reranker if available, otherwise fall back to Cohere
        if self.gpt_reranker:
            reranked_data = self.gpt_reranker.rerank_snippets(
                rerank_query, all_snippets, self.top_k_snippets
            )
        else:
            reranked_data = self.cohere_client.rerank_snippets(
                rerank_query, all_snippets, self.top_k_snippets
            )

        if not reranked_data:
            logger.warning(f"Reranking produced no snippets for {arxiv_id}. Skipping LLM analysis.")
            return {"jwstscience": 0.0, "quotes": [], 
                   "reason": "Keyword snippets found but none survived reranking/filtering."}
                   
        # Check reranker threshold
        top_score = reranked_data[0].get('score')
        if top_score is not None and top_score < self.reranker_threshold:
            logger.info(f"Skipping LLM science analysis for {arxiv_id}: Top reranker score ({top_score:g}) below threshold ({self.reranker_threshold}).")
            return {
                "jwstscience": 0.0, 
                "quotes": [],
                "reason": f"Skipped LLM analysis: Top reranker score ({top_score:g}) was below the threshold ({self.reranker_threshold}).",
            }

        # Prepare LLM input
        reranked_snippets_for_llm = [item['snippet'] for item in reranked_data]
        snippets_text = "\n---\n".join([f"Excerpt {i+1}:\n{s}" for i, s in enumerate(reranked_snippets_for_llm)])
        max_chars = 30000 
        if len(snippets_text) > max_chars:
            logger.warning(f"Total snippet text for {arxiv_id} exceeds {max_chars} chars, truncating.")
            snippets_text = snippets_text[:max_chars]

        system_prompt = self.prompts.get('science_system')
        user_prompt_template = self.prompts.get('science_user')
        if not system_prompt or not user_prompt_template:
            logger.error(f"Science prompts not loaded correctly for {arxiv_id}. Check prompts directory.")
            return {"jwstscience": -1.0, "reason": "Analysis failed: Prompts missing", 
                   "quotes": [], "error": "prompt_missing"}
        
        try:
            user_prompt = user_prompt_template.format(snippets_text=snippets_text)
        except KeyError:
            logger.error("Failed to format science user prompt - missing '{snippets_text}' placeholder?")
            return {"jwstscience": -1.0, "reason": "Analysis failed: Prompt formatting error", 
                   "quotes": [], "error": "prompt_format_error"}

        # Call LLM
        llm_result = self.openai_client.call_parse(
            system_prompt, user_prompt, JWSTScienceLabelerModel
        )

        if llm_result is None or "error" in llm_result:
            error_reason = f"LLM analysis failed: {llm_result.get('message', 'Unknown error') if llm_result else 'Unknown error'}"
            error_type = llm_result.get('error', 'unknown') if llm_result else 'unknown'
            return {"jwstscience": -1.0, "reason": error_reason, "quotes": [], "error": error_type}

        return llm_result