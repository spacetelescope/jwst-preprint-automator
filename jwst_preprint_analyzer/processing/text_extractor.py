"""Text extraction and snippet processing."""

import logging
import re
from typing import List, Set

import nltk
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


class TextExtractor:
    """Handles text extraction and snippet generation."""
    
    def __init__(self, context_sentences: int = 3):
        self.context_sentences = context_sentences
        self._ensure_nltk_data()
        
    def _ensure_nltk_data(self):
        """Ensure NLTK punkt tokenizer is available."""
        try:
            nltk.data.find('tokenizers/punkt')
        except (nltk.downloader.DownloadError, LookupError):
            logger.info("NLTK 'punkt' tokenizer not found. Attempting download...")
            try:
                nltk.download('punkt', quiet=True)
                nltk.data.find('tokenizers/punkt')
                logger.info("'punkt' tokenizer downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download NLTK 'punkt' tokenizer: {e}")
                logger.error("Please install it manually: run `python -c \"import nltk; nltk.download('punkt')\"`")
                raise
    
    def extract_relevant_snippets(self, paper_text: str, keywords: List[str]) -> List[str]:
        """Extract sentences containing keywords, plus surrounding context sentences."""
        if not paper_text:
            return []

        # Preprocess text
        paper_text = re.sub(r'\n{2,}', '\n\n', paper_text)
        paper_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', paper_text)
        paper_text = re.sub(r' {2,}', ' ', paper_text)

        try:
            sentences = sent_tokenize(paper_text)
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed: {e}. Falling back to simple split.")
            sentences = [line for line in paper_text.splitlines() if line.strip()]

        if not sentences:
            return []

        relevant_indices: Set[int] = set()
        extracted_snippets: Set[str] = set() 

        # Find indices
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            for keyword in keywords: 
                if keyword in sentence_lower:
                    relevant_indices.add(i)
                    break

        # Expand context
        final_indices: Set[int] = set()
        num_sentences = len(sentences) 
        for i in relevant_indices:
            start = max(0, i - self.context_sentences)
            end = min(num_sentences, i + self.context_sentences + 1)
            final_indices.update(range(start, end))

        # Create snippets 
        if not final_indices:
            return []

        sorted_indices = sorted(list(final_indices))
        current_snippet_sentences = []
        last_index = -2

        for index in sorted_indices:
            if index != last_index + 1 and current_snippet_sentences:
                extracted_snippets.add(" ".join(current_snippet_sentences).strip())
                current_snippet_sentences = []

            current_snippet_sentences.append(sentences[index])
            last_index = index

        if current_snippet_sentences:
            extracted_snippets.add(" ".join(current_snippet_sentences).strip())

        return [s for s in list(extracted_snippets) if len(s) > 10]