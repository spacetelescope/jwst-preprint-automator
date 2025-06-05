"""JWST DOI analysis."""

import logging
from pathlib import Path
from typing import Dict, List

from ..models import JWSTDOILabelerModel
from ..clients.openai import OpenAIClient
from ..clients.cohere import CohereClient
from ..processing.text_extractor import TextExtractor
from .science import ScienceAnalyzer

logger = logging.getLogger(__name__)


class DOIAnalyzer:
    """Analyzes papers for JWST DOIs."""
    
    DOI_KEYWORDS = [
        "mast", "mast archive",
        "10.17909",  # Specific prefix for MAST
        "data availability", "acknowledgments",
        "program id", "proposal id",
    ]
    DOI_KEYWORDS_LOWER = sorted(
        [k.lower() for k in DOI_KEYWORDS], 
        key=len, reverse=True
    )
    
    def __init__(self, 
                 openai_client: OpenAIClient,
                 cohere_client: CohereClient,
                 text_extractor: TextExtractor,
                 prompts: Dict[str, str],
                 top_k_snippets: int = 15,
                 validate_llm: bool = False):
        self.openai_client = openai_client
        self.cohere_client = cohere_client
        self.text_extractor = text_extractor
        self.prompts = prompts
        self.top_k_snippets = top_k_snippets
        self.validate_llm = validate_llm
        
    def analyze(self, arxiv_id: str, text_path: Path) -> Dict:
        """Analyze paper for JWST DOIs using extracted snippets."""
        logger.info(f"Analyzing JWST DOIs for {arxiv_id}")
        
        if not text_path.exists():
            logger.warning(f"Text file not found for {arxiv_id}, cannot analyze DOI.")
            return {"jwstdoi": -1.0, "reason": "Analysis failed: Text file missing", 
                   "quotes": [], "error": "missing_text_file"}

        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                paper_text = f.read()
        except Exception as e:
            logger.error(f"Failed to read text file {text_path}: {e}")
            return {"jwstdoi": -1.0, "reason": "Analysis failed: Cannot read text file", 
                   "quotes": [], "error": "read_error"}

        # Extract snippets (combine DOI and science keywords)
        keywords_to_find = self.DOI_KEYWORDS_LOWER + [
            k for k in ScienceAnalyzer.SCIENCE_KEYWORDS_LOWER 
            if k not in self.DOI_KEYWORDS_LOWER
        ]
        all_snippets = self.text_extractor.extract_relevant_snippets(
            paper_text, keywords_to_find
        )

        if not all_snippets:
            logger.info(f"No relevant keywords found for DOI analysis in {arxiv_id}.")
            return {"jwstdoi": 0.0, "quotes": [], 
                   "reason": "No relevant keywords (e.g., DOI, 10.17909, data availability, JWST) found in text."}

        # Rerank snippets
        rerank_query = self.prompts.get('rerank_doi_query')
        if not rerank_query:
            logger.error("Rerank DOI query prompt ('rerank_doi_query.txt') not found or empty.")
            return {"jwstdoi": -1.0, "reason": "Analysis failed: Missing rerank DOI query prompt", 
                   "quotes": [], "error": "prompt_missing"}
                   
        reranked_data = self.cohere_client.rerank_snippets(
            rerank_query, all_snippets, self.top_k_snippets
        )

        if not reranked_data:
            logger.warning(f"Reranking produced no snippets for DOI analysis {arxiv_id}. Skipping LLM.")
            return {"jwstdoi": 0.0, "quotes": [], 
                   "reason": "Keyword snippets found but none survived reranking/filtering for DOI check."}

        # Prepare LLM input
        reranked_snippets_for_llm = [item['snippet'] for item in reranked_data]
        snippets_text = "\n---\n".join([f"Excerpt {i+1}:\n{s}" for i, s in enumerate(reranked_snippets_for_llm)])
        max_chars = 30000 
        if len(snippets_text) > max_chars:
            logger.warning(f"Total snippet text for DOI analysis {arxiv_id} exceeds {max_chars} chars, truncating.")
            snippets_text = snippets_text[:max_chars]

        system_prompt = self.prompts.get('doi_system')
        user_prompt_template = self.prompts.get('doi_user')
        if not system_prompt or not user_prompt_template:
            logger.error(f"DOI prompts not loaded correctly for {arxiv_id}. Check prompts directory.")
            return {"jwstdoi": -1.0, "reason": "Analysis failed: Prompts missing", 
                   "quotes": [], "error": "prompt_missing"}
                   
        try:
            user_prompt = user_prompt_template.format(snippets_text=snippets_text)
        except KeyError:
            logger.error("Failed to format DOI user prompt - missing '{snippets_text}' placeholder?")
            return {"jwstdoi": -1.0, "reason": "Analysis failed: Prompt formatting error", 
                   "quotes": [], "error": "prompt_format_error"}

        # Call LLM
        llm_result = self.openai_client.call_parse(
            system_prompt, user_prompt, JWSTDOILabelerModel
        )

        if llm_result is None or "error" in llm_result:
            error_reason = f"LLM analysis failed for DOI: {llm_result.get('message', 'Unknown error') if llm_result else 'Unknown error'}"
            error_type = llm_result.get('error', 'unknown') if llm_result else 'unknown'
            logger.warning(f"DOI analysis failed for {arxiv_id}: {error_reason}")
            return {"jwstdoi": -1.0, "reason": error_reason, "quotes": [], "error": error_type}

        return llm_result