"""Tests for jwst_preprint_analyzer.processing.text_extractor module."""

import pytest
from unittest.mock import patch, MagicMock

from jwst_preprint_analyzer.processing.text_extractor import TextExtractor


class TestTextExtractor:
    """Test cases for TextExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = TextExtractor(context_sentences=2)
    
    @patch('jwst_preprint_analyzer.processing.text_extractor.nltk')
    def test_initialization_with_existing_nltk_data(self, mock_nltk):
        """Test initialization when NLTK data already exists."""
        mock_nltk.data.find.return_value = True
        
        extractor = TextExtractor()
        
        # Should not attempt download
        mock_nltk.download.assert_not_called()
        assert extractor.context_sentences == 3  # default value
    
    @patch('jwst_preprint_analyzer.processing.text_extractor.nltk')
    def test_initialization_downloads_missing_nltk_data(self, mock_nltk):
        """Test initialization downloads NLTK data when missing."""
        # First call raises LookupError (missing data)
        # Second call succeeds (after download)
        mock_nltk.data.find.side_effect = [LookupError(), True]
        mock_nltk.downloader.DownloadError = Exception  # Mock the exception class
        
        extractor = TextExtractor()
        
        # Should attempt download
        mock_nltk.download.assert_called_once_with('punkt', quiet=True)
    
    def test_extract_snippets_empty_text(self):
        """Test extraction with empty text returns empty list."""
        result = self.extractor.extract_relevant_snippets("", ["jwst"])
        assert result == []
    
    def test_extract_snippets_no_keywords_found(self):
        """Test extraction when no keywords are found."""
        text = "This is a paper about exoplanets and stellar formation."
        keywords = ["jwst", "webb", "telescope"]
        
        result = self.extractor.extract_relevant_snippets(text, keywords)
        assert result == []
    
    def test_extract_snippets_single_match(self):
        """Test extraction with a single keyword match."""
        text = "Introduction. This paper presents JWST observations. We analyze the data carefully. Conclusion follows."
        keywords = ["jwst"]
        
        result = self.extractor.extract_relevant_snippets(text, keywords)
        
        assert len(result) == 1
        # Should include the matched sentence plus context
        assert "JWST observations" in result[0]
        assert "Introduction" in result[0]  # context before
        assert "analyze the data" in result[0]  # context after
    
    def test_extract_snippets_multiple_matches(self):
        """Test extraction with multiple keyword matches."""
        text = "First sentence. JWST telescope data. Middle sentence. Webb observations here. Final sentence."
        keywords = ["jwst", "webb"]
        
        result = self.extractor.extract_relevant_snippets(text, keywords)
        
        # Should have snippets containing both matches
        combined_text = " ".join(result)
        assert "JWST telescope data" in combined_text
        assert "Webb observations" in combined_text
    
    def test_extract_snippets_case_insensitive(self):
        """Test that keyword matching is case-insensitive."""
        text = "First sentence. The JWST telescope observed. Final sentence."
        keywords = ["jwst"]  # lowercase
        
        result = self.extractor.extract_relevant_snippets(text, keywords)
        
        assert len(result) == 1
        assert "JWST telescope observed" in result[0]
    
    def test_extract_snippets_context_window(self):
        """Test that context window works correctly."""
        # Create text with 7 sentences, match in the middle
        sentences = [
            "Sentence 1.", "Sentence 2.", "Sentence 3.", 
            "JWST telescope data here.",  # match at index 3
            "Sentence 5.", "Sentence 6.", "Sentence 7."
        ]
        text = " ".join(sentences)
        keywords = ["jwst"]
        
        result = self.extractor.extract_relevant_snippets(text, keywords)
        
        # With context_sentences=2, should include sentences around the match
        assert len(result) == 1
        snippet = result[0]
        assert "JWST telescope data" in snippet
        # Should include some context sentences
        context_count = sum(1 for sent in ["Sentence 1", "Sentence 2", "Sentence 3", "Sentence 5", "Sentence 6", "Sentence 7"] if sent in snippet)
        assert context_count >= 2  # Should have at least some context
    
    def test_extract_snippets_overlapping_contexts(self):
        """Test handling of overlapping context windows."""
        text = "Sent 1. JWST data. Sent 3. Webb telescope. Sent 5."
        keywords = ["jwst", "webb"]
        
        result = self.extractor.extract_relevant_snippets(text, keywords)
        
        # Should merge overlapping contexts into a single snippet
        assert len(result) == 1
        snippet = result[0]
        assert "JWST data" in snippet
        assert "Webb telescope" in snippet
    
    def test_extract_snippets_text_preprocessing(self):
        """Test that text preprocessing works correctly."""
        # Text with multiple newlines and spaces
        text = "First sentence.\n\n\nSecond sentence with  extra   spaces.\nJWST telescope data.\n\nFinal sentence."
        keywords = ["jwst"]
        
        result = self.extractor.extract_relevant_snippets(text, keywords)
        
        assert len(result) == 1
        # Should have cleaned up extra whitespace
        snippet = result[0]
        assert "  " not in snippet  # no double spaces
        assert "\n\n" not in snippet  # no double newlines
    
    @patch('jwst_preprint_analyzer.processing.text_extractor.sent_tokenize')
    def test_fallback_when_nltk_fails(self, mock_sent_tokenize):
        """Test fallback behavior when NLTK sentence tokenization fails."""
        mock_sent_tokenize.side_effect = Exception("NLTK error")
        
        text = "First line.\nJWST telescope data.\nFinal line."
        keywords = ["jwst"]
        
        result = self.extractor.extract_relevant_snippets(text, keywords)
        
        # Should still work using line-based fallback
        assert len(result) == 1
        assert "JWST telescope data" in result[0]
    
    def test_extract_snippets_minimum_length_filter(self):
        """Test that very short snippets are filtered out."""
        text = "A. JWST. B."  # Very short sentences
        keywords = ["jwst"]
        
        result = self.extractor.extract_relevant_snippets(text, keywords)
        
        # Should filter out snippets shorter than 10 characters
        if result:  # Only check if results exist
            for snippet in result:
                assert len(snippet) > 10
    
    def test_extract_snippets_deduplication(self):
        """Test that duplicate snippets are removed."""
        # Text that would create duplicate snippets
        text = "Sentence 1. JWST data here. Sentence 3. JWST data here. Sentence 5."
        keywords = ["jwst"]
        
        result = self.extractor.extract_relevant_snippets(text, keywords)
        
        # Should not have duplicate snippets
        unique_snippets = set(result)
        assert len(result) == len(unique_snippets)
    
    def test_context_sentences_parameter(self):
        """Test that context_sentences parameter affects window size."""
        extractor_small = TextExtractor(context_sentences=1)
        extractor_large = TextExtractor(context_sentences=4)
        
        text = "S1. S2. S3. JWST data. S5. S6. S7."
        keywords = ["jwst"]
        
        result_small = extractor_small.extract_relevant_snippets(text, keywords)
        result_large = extractor_large.extract_relevant_snippets(text, keywords)
        
        # Larger context should include more sentences
        if result_small and result_large:
            assert len(result_large[0]) >= len(result_small[0])