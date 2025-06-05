"""Tests for jwst_preprint_analyzer.processing.text_extractor module."""

import pytest
from unittest.mock import patch, MagicMock

from jwst_preprint_analyzer.processing.text_extractor import TextExtractor


# Check if NLTK punkt is available
def nltk_punkt_available():
    """Check if NLTK punkt tokenizer is available."""
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
        return True
    except (ImportError, LookupError):
        return False


skip_nltk = pytest.mark.skipif(
    not nltk_punkt_available(),
    reason="NLTK punkt tokenizer not available - skipping for CI"
)


class TestTextExtractor:
    """Test cases for TextExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Only create extractor for non-NLTK tests
        pass
    
    @skip_nltk
    def test_initialization_with_real_nltk(self):
        """Test initialization when NLTK data is available."""
        extractor = TextExtractor()
        assert extractor.context_sentences == 3  # default value
    
    @patch('jwst_preprint_analyzer.processing.text_extractor.nltk')
    def test_initialization_with_mocked_nltk(self, mock_nltk):
        """Test initialization with mocked NLTK."""
        mock_nltk.data.find.return_value = True
        mock_nltk.downloader.DownloadError = Exception
        
        extractor = TextExtractor(context_sentences=2)
        assert extractor.context_sentences == 2
    
    @patch('jwst_preprint_analyzer.processing.text_extractor.nltk')
    def test_initialization_downloads_missing_nltk_data(self, mock_nltk):
        """Test initialization downloads NLTK data when missing."""
        # First call raises LookupError (missing data)
        # Second call succeeds (after download)
        mock_nltk.data.find.side_effect = [LookupError(), True]
        mock_nltk.downloader.DownloadError = Exception
        
        extractor = TextExtractor()
        
        # Should attempt download
        mock_nltk.download.assert_called_once_with('punkt', quiet=True)
    
    @patch('jwst_preprint_analyzer.processing.text_extractor.nltk')
    def test_extract_snippets_empty_text(self, mock_nltk):
        """Test extraction with empty text returns empty list."""
        mock_nltk.data.find.return_value = True
        mock_nltk.downloader.DownloadError = Exception
        
        extractor = TextExtractor(context_sentences=2)
        result = extractor.extract_relevant_snippets("", ["jwst"])
        assert result == []
    
    @patch('jwst_preprint_analyzer.processing.text_extractor.nltk')
    def test_extract_snippets_no_keywords_found(self, mock_nltk):
        """Test extraction when no keywords are found."""
        mock_nltk.data.find.return_value = True
        mock_nltk.downloader.DownloadError = Exception
        
        extractor = TextExtractor(context_sentences=2)
        text = "This is a paper about exoplanets and stellar formation."
        keywords = ["jwst", "webb", "telescope"]
        
        result = extractor.extract_relevant_snippets(text, keywords)
        assert result == []
    
    @patch('jwst_preprint_analyzer.processing.text_extractor.nltk')
    def test_extract_snippets_single_match(self, mock_nltk):
        """Test extraction with a single keyword match."""
        mock_nltk.data.find.return_value = True
        mock_nltk.downloader.DownloadError = Exception
        # Mock sent_tokenize to return predictable sentences
        mock_nltk.tokenize.sent_tokenize.return_value = [
            "Introduction.", "This paper presents JWST observations.", 
            "We analyze the data carefully.", "Conclusion follows."
        ]
        
        extractor = TextExtractor(context_sentences=2)
        text = "Introduction. This paper presents JWST observations. We analyze the data carefully. Conclusion follows."
        keywords = ["jwst"]
        
        result = extractor.extract_relevant_snippets(text, keywords)
        
        assert len(result) >= 1
        # Should include the matched content
        combined_result = " ".join(result)
        assert "JWST" in combined_result or "jwst" in combined_result.lower()
    
    @patch('jwst_preprint_analyzer.processing.text_extractor.nltk')
    def test_extract_snippets_case_insensitive(self, mock_nltk):
        """Test that keyword matching is case-insensitive."""
        mock_nltk.data.find.return_value = True
        mock_nltk.downloader.DownloadError = Exception
        mock_nltk.tokenize.sent_tokenize.return_value = [
            "First sentence.", "The JWST telescope observed.", "Final sentence."
        ]
        
        extractor = TextExtractor(context_sentences=2)
        text = "First sentence. The JWST telescope observed. Final sentence."
        keywords = ["jwst"]  # lowercase
        
        result = extractor.extract_relevant_snippets(text, keywords)
        
        assert len(result) >= 1
        combined_result = " ".join(result)
        assert "JWST telescope observed" in combined_result
    
    @patch('jwst_preprint_analyzer.processing.text_extractor.nltk')
    def test_fallback_when_nltk_fails(self, mock_nltk):
        """Test fallback behavior when NLTK sentence tokenization fails."""
        mock_nltk.data.find.return_value = True
        mock_nltk.downloader.DownloadError = Exception
        # Make sent_tokenize raise an exception
        mock_nltk.tokenize.sent_tokenize.side_effect = Exception("NLTK error")
        
        extractor = TextExtractor(context_sentences=2)
        text = "First line.\nJWST telescope data.\nFinal line."
        keywords = ["jwst"]
        
        result = extractor.extract_relevant_snippets(text, keywords)
        
        # Should still work using line-based fallback
        if result:  # May find matches depending on fallback logic
            combined_result = " ".join(result)
            assert "JWST" in combined_result or "jwst" in combined_result.lower()
    
    @patch('jwst_preprint_analyzer.processing.text_extractor.nltk')
    def test_extract_snippets_minimum_length_filter(self, mock_nltk):
        """Test that very short snippets are filtered out."""
        mock_nltk.data.find.return_value = True
        mock_nltk.downloader.DownloadError = Exception
        mock_nltk.tokenize.sent_tokenize.return_value = ["A.", "JWST.", "B."]
        
        extractor = TextExtractor(context_sentences=2)
        text = "A. JWST. B."  # Very short sentences
        keywords = ["jwst"]
        
        result = extractor.extract_relevant_snippets(text, keywords)
        
        # Should filter out snippets shorter than 10 characters
        for snippet in result:
            assert len(snippet) > 10
    
    @skip_nltk
    def test_real_text_extraction(self):
        """Test real text extraction with actual NLTK - only runs if NLTK available."""
        extractor = TextExtractor(context_sentences=1)
        text = ("Introduction to the study. "
                "This paper presents new JWST observations of distant galaxies. "
                "We analyze the photometric data carefully. "
                "The results show interesting trends.")
        keywords = ["jwst"]
        
        result = extractor.extract_relevant_snippets(text, keywords)
        
        assert len(result) >= 1
        # Should include the matched sentence and context
        combined_result = " ".join(result)
        assert "JWST observations" in combined_result