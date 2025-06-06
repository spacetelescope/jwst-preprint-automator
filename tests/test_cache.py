"""Tests for jwst_preprint_analyzer.utils.cache module."""

import json
import tempfile
import pytest
from pathlib import Path

from jwst_preprint_analyzer.utils.cache import load_cache, save_cache


class TestLoadCache:
    """Test cases for load_cache function."""
    
    def test_load_existing_cache(self):
        """Test loading an existing cache file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
            json.dump(test_data, f)
            cache_file = Path(f.name)
        
        try:
            result = load_cache(cache_file)
            assert result == test_data
        finally:
            cache_file.unlink()
    
    def test_load_nonexistent_cache(self):
        """Test loading a cache file that doesn't exist."""
        nonexistent_file = Path("/tmp/nonexistent_cache.json")
        result = load_cache(nonexistent_file)
        assert result == {}
    
    def test_load_corrupted_cache(self):
        """Test loading a corrupted JSON cache file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            cache_file = Path(f.name)
        
        try:
            result = load_cache(cache_file)
            assert result == {}  # Should return empty dict for corrupted file
        finally:
            cache_file.unlink()
    
    def test_load_empty_cache(self):
        """Test loading an empty cache file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            cache_file = Path(f.name)
        
        try:
            result = load_cache(cache_file)
            assert result == {}
        finally:
            cache_file.unlink()
    
    def test_load_cache_with_unicode(self):
        """Test loading cache with Unicode characters."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            test_data = {"unicode_key": "café 🔭 émoji", "paper_title": "Analysis of α Centauri"}
            json.dump(test_data, f, ensure_ascii=False)
            cache_file = Path(f.name)
        
        try:
            result = load_cache(cache_file)
            assert result == test_data
            assert "🔭" in result["unicode_key"]
        finally:
            cache_file.unlink()


class TestSaveCache:
    """Test cases for save_cache function."""
    
    def test_save_new_cache(self):
        """Test saving data to a new cache file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "test_cache.json"
            test_data = {"arxiv_id": "2309.16028", "processed": True, "score": 0.85}
            
            save_cache(cache_file, test_data)
            
            # Verify file was created and contains correct data
            assert cache_file.exists()
            with open(cache_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            assert loaded_data == test_data
    
    def test_save_overwrites_existing_cache(self):
        """Test that saving overwrites existing cache file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "test_cache.json"
            
            # Save initial data
            initial_data = {"old_key": "old_value"}
            save_cache(cache_file, initial_data)
            
            # Save new data
            new_data = {"new_key": "new_value"}
            save_cache(cache_file, new_data)
            
            # Verify new data replaced old data
            with open(cache_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            assert loaded_data == new_data
            assert "old_key" not in loaded_data
    
    def test_save_empty_cache(self):
        """Test saving an empty cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "empty_cache.json"
            empty_data = {}
            
            save_cache(cache_file, empty_data)
            
            assert cache_file.exists()
            with open(cache_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            assert loaded_data == {}
    
    def test_save_cache_with_unicode(self):
        """Test saving cache with Unicode characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "unicode_cache.json"
            test_data = {"title": "Étude de l'univers 🌌", "author": "José García"}
            
            save_cache(cache_file, test_data)
            
            # Read file and verify Unicode was preserved
            with open(cache_file, 'r', encoding='utf-8') as f:
                content = f.read()
                loaded_data = json.loads(content)
            
            assert loaded_data == test_data
            assert "🌌" in loaded_data["title"]
    
    def test_save_cache_with_complex_data(self):
        """Test saving cache with complex nested data structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "complex_cache.json"
            test_data = {
                "papers": {
                    "2309.16028": {
                        "title": "JWST Observations",
                        "scores": [0.1, 0.5, 0.9],
                        "metadata": {
                            "processed": True,
                            "errors": None
                        }
                    }
                },
                "settings": {
                    "threshold": 0.7,
                    "enabled": True
                }
            }
            
            save_cache(cache_file, test_data)
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            assert loaded_data == test_data
    
    def test_save_cache_with_nonexistent_parent_directory(self):
        """Test that save_cache handles missing parent directories gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested path that doesn't exist
            cache_file = Path(temp_dir) / "nested" / "deep" / "cache.json"
            test_data = {"test": "data"}
            
            # Should not raise error, but also shouldn't create the file
            save_cache(cache_file, test_data)
            assert not cache_file.exists()  # File should not have been created


class TestCacheIntegration:
    """Integration tests for cache load/save operations."""
    
    def test_save_and_load_roundtrip(self):
        """Test that data saved and then loaded remains identical."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "roundtrip_cache.json"
            
            original_data = {
                "arxiv_papers": ["2309.16028", "2310.12345"],
                "processing_stats": {
                    "processed": 42,
                    "skipped": 3,
                    "errors": 1
                },
                "scores": [0.1, 0.5, 0.9, 0.2],
                "unicode_test": "café 🚀"
            }
            
            # Save and load
            save_cache(cache_file, original_data)
            loaded_data = load_cache(cache_file)
            
            assert loaded_data == original_data
    
    def test_multiple_save_load_cycles(self):
        """Test multiple save/load cycles maintain data integrity."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = Path(temp_dir) / "cycle_cache.json"
            
            # Multiple cycles with different data
            test_datasets = [
                {"cycle": 1, "data": [1, 2, 3]},
                {"cycle": 2, "data": {"nested": "object"}},
                {"cycle": 3, "data": "simple string"}
            ]
            
            for data in test_datasets:
                save_cache(cache_file, data)
                loaded = load_cache(cache_file)
                assert loaded == data