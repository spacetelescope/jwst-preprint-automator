"""GPT-4.1-nano reranker client for replacing Cohere reranking."""

import logging
import math
from typing import List, Dict, Union, Optional

from .openai import OpenAIClient

logger = logging.getLogger(__name__)

# YES_TOKEN_ID = 9642  # "Yes" token in `cl100k_base`
# NO_TOKEN_ID = 2822   # "No" token in `cl100k_base`

YES_TOKEN_ID = 13022 # "Yes" token in `o200k_base` 
NO_TOKEN_ID = 3160   # "No" token in `o200k_base`  

class GPTReranker:
    """GPT-4.1-nano based reranker using logit bias and log probabilities."""
    
    def __init__(self, openai_client: OpenAIClient, model: str = 'gpt-4.1-nano-2025-04-14'):
        self.openai_client = openai_client
        self.model = model
        self.yes_token_id = YES_TOKEN_ID
        self.no_token_id = NO_TOKEN_ID
    
    def _create_reranking_prompt(self, query: str, snippet: str) -> List[Dict[str, str]]:
        """Create a prompt for yes/no reranking."""
        system_prompt = (
            "You are a relevance classifier. Answer only 'Yes' or 'No' based on whether "
            "the given text snippet is relevant to the query. Do not provide explanations."
        )
        
        user_prompt = f"""Query: {query}

Text snippet:
{snippet}

Is this text snippet relevant to the query? Answer only 'Yes' or 'No'."""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _extract_yes_probability(self, logprobs_data) -> float:
        """Extract the probability of 'Yes' from log probabilities."""
        try:
            if not logprobs_data or not logprobs_data.content:
                return 0.0
            
            # Get the first token's log probabilities
            first_token_logprobs = logprobs_data.content[0]
            
            # Look for 'Yes' token in top logprobs
            for token_info in first_token_logprobs.top_logprobs:
                if token_info.token.strip().lower() in ['yes', 'y']:
                    # Convert log probability to probability
                    return math.exp(token_info.logprob)
            
            # If 'Yes' not found in top logprobs, return 0
            return 0.0
            
        except Exception as e:
            logger.error(f"Error extracting Yes probability: {e}")
            return 0.0
    
    def rerank_snippets(self, query: str, snippets: List[str], 
                       top_k: int) -> List[Dict[str, Union[str, Optional[float]]]]:
        """Rerank snippets using GPT-4.1-nano with logit bias and log probabilities."""
        if not snippets:
            return []
        
        logger.debug(f"Reranking {len(snippets)} snippets with GPT model '{self.model}' for query: '{query}'")
        
        # Filter out empty snippets
        non_empty_snippets = [s for s in snippets if s and s.strip()]
        if not non_empty_snippets:
            logger.warning("All extracted snippets were empty after stripping.")
            return []
        
        scored_snippets = []
        
        # Process each snippet individually
        for snippet in non_empty_snippets:
            try:
                # Create prompt for this snippet
                messages = self._create_reranking_prompt(query, snippet)
                
                # Set up logit bias to constrain to Yes/No
                logit_bias = {}
                # Bias towards Yes/No tokens and against others
                logit_bias[self.yes_token_id] = 0
                logit_bias[self.no_token_id] = 0
                
                # Make API call with logit bias and log probabilities
                response = self.openai_client.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1,
                    temperature=0.4,
                    logprobs=True,
                    top_logprobs=2,
                    logit_bias=logit_bias if logit_bias else None
                )
                
                # Extract probability score
                score = self._extract_yes_probability(response.choices[0].logprobs)
                scored_snippets.append({'snippet': snippet, 'score': score})
                
            except Exception as e:
                logger.error(f"Error processing snippet for reranking: {e}")
                scored_snippets.append({'snippet': snippet, 'score': 0.0})
        
        # Sort by score (highest first) and return top_k
        scored_snippets.sort(key=lambda x: x['score'] or 0, reverse=True)
        result = scored_snippets[:top_k]
        
        if result:
            top_scores = [f"{r['score']:.3f}" if r['score'] is not None else "None" for r in result[:3]]
            logger.info(f"GPT reranked scores (top 3): {top_scores}")
        
        return result