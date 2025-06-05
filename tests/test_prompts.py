"""Tests for jwst_preprint_analyzer.utils.prompts module."""

import tempfile
import pytest
from pathlib import Path

from jwst_preprint_analyzer.utils.prompts import load_prompts, PROMPT_FILES


class TestLoadPrompts:
    """Test cases for the load_prompts function."""
    
    def test_load_existing_prompts(self):
        """Test loading prompts from existing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_dir = Path(temp_dir)
            
            # Create all essential prompt files
            test_prompts = {
                'science_system.txt': 'You are a science classifier.',
                'science_user.txt': 'Classify this paper: {text}',
                'rerank_science_query.txt': 'JWST telescope observations',
                'doi_system.txt': 'You are a DOI classifier.',
                'doi_user.txt': 'Find DOIs in this text: {text}',
                'rerank_doi_query.txt': 'JWST DOI 10.17909'
            }
            
            for filename, content in test_prompts.items():
                (prompts_dir / filename).write_text(content, encoding='utf-8')
            
            loaded = load_prompts(prompts_dir)
            
            # Check that essential prompts are loaded
            assert 'science_system' in loaded
            assert 'science_user' in loaded 
            assert 'rerank_science_query' in loaded
            assert loaded['science_system'] == 'You are a science classifier.'
            assert loaded['science_user'] == 'Classify this paper: {text}'
            assert loaded['rerank_science_query'] == 'JWST telescope observations'
    
    def test_missing_essential_prompt_raises_error(self):
        """Test that missing essential prompt files raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_dir = Path(temp_dir)
            
            # Don't create any files
            with pytest.raises(FileNotFoundError) as exc_info:
                load_prompts(prompts_dir)
            
            assert "Essential prompt file missing" in str(exc_info.value)
    
    def test_missing_optional_prompt_continues(self):
        """Test that missing optional prompt files are handled gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_dir = Path(temp_dir)
            
            # Create only essential prompt files
            essential_files = [
                'science_system.txt', 'science_user.txt', 'rerank_science_query.txt',
                'doi_system.txt', 'doi_user.txt', 'rerank_doi_query.txt'
            ]
            
            for filename in essential_files:
                (prompts_dir / filename).write_text(f"Content for {filename}", encoding='utf-8')
            
            loaded = load_prompts(prompts_dir)
            
            # Essential prompts should be loaded
            assert loaded['science_system'] == "Content for science_system.txt"
            
            # Optional prompts should be empty strings
            assert loaded['science_validate_system'] == ""
            assert loaded['science_validate_user'] == ""
    
    def test_nonexistent_directory_creation(self):
        """Test that function creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_dir = Path(temp_dir) / "nonexistent" / "prompts"
            
            # This should create the directory but then fail due to missing files
            with pytest.raises(FileNotFoundError):
                load_prompts(prompts_dir)
            
            # Directory should have been created
            assert prompts_dir.exists()
            assert prompts_dir.is_dir()
    
    def test_prompt_files_constant(self):
        """Test that PROMPT_FILES constant contains expected keys."""
        expected_keys = {
            'science_system', 'science_user', 'science_validate_system', 'science_validate_user',
            'rerank_science_query', 'doi_system', 'doi_user', 'doi_validate_system',
            'doi_validate_user', 'rerank_doi_query'
        }
        
        assert set(PROMPT_FILES.keys()) == expected_keys
        
        # Check that all values end with .txt
        for filename in PROMPT_FILES.values():
            assert filename.endswith('.txt')
    
    def test_file_encoding_handling(self):
        """Test that function handles UTF-8 encoding correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_dir = Path(temp_dir)
            
            # Create file with UTF-8 content including special characters
            test_content = "You are a classifier. Use émojis: 🔭 🌌"
            (prompts_dir / 'science_system.txt').write_text(test_content, encoding='utf-8')
            
            # Create minimal essential files
            essential_files = ['science_user.txt', 'rerank_science_query.txt',
                             'doi_system.txt', 'doi_user.txt', 'rerank_doi_query.txt']
            for filename in essential_files:
                (prompts_dir / filename).write_text("minimal", encoding='utf-8')
            
            loaded = load_prompts(prompts_dir)
            
            assert loaded['science_system'] == test_content
            assert "🔭" in loaded['science_system']
    
    def test_empty_prompt_files(self):
        """Test handling of empty prompt files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_dir = Path(temp_dir)
            
            # Create empty essential files
            essential_files = [
                'science_system.txt', 'science_user.txt', 'rerank_science_query.txt',
                'doi_system.txt', 'doi_user.txt', 'rerank_doi_query.txt'
            ]
            
            for filename in essential_files:
                (prompts_dir / filename).write_text("", encoding='utf-8')
            
            loaded = load_prompts(prompts_dir)
            
            # Should load empty strings without error
            for key in ['science_system', 'science_user', 'doi_system']:
                assert loaded[key] == ""