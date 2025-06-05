"""Tests for jwst_preprint_analyzer.models module."""

import pytest
from pydantic import ValidationError

from jwst_preprint_analyzer.models import JWSTScienceLabelerModel, JWSTDOILabelerModel


class TestJWSTScienceLabelerModel:
    """Test cases for JWSTScienceLabelerModel."""
    
    def test_valid_model_creation(self):
        """Test creating a valid JWSTScienceLabelerModel instance."""
        data = {
            "quotes": ["This paper uses JWST observations", "We analyze NIRCam data"],
            "jwstscience": 0.9,
            "reason": "Paper presents clear JWST science content"
        }
        model = JWSTScienceLabelerModel(**data)
        
        assert model.quotes == data["quotes"]
        assert model.jwstscience == data["jwstscience"]
        assert model.reason == data["reason"]
    
    def test_score_validation_range(self):
        """Test that jwstscience score validation works correctly."""
        # Valid scores (0.0 to 1.0)
        valid_data = {
            "quotes": ["Sample quote"],
            "jwstscience": 0.5,
            "reason": "Sample reason"
        }
        model = JWSTScienceLabelerModel(**valid_data)
        assert model.jwstscience == 0.5
        
        # Edge cases
        edge_cases = [0.0, 1.0]
        for score in edge_cases:
            valid_data["jwstscience"] = score
            model = JWSTScienceLabelerModel(**valid_data)
            assert model.jwstscience == score
    
    def test_empty_quotes_list(self):
        """Test model creation with empty quotes list."""
        data = {
            "quotes": [],
            "jwstscience": 0.1,
            "reason": "No supporting quotes found"
        }
        model = JWSTScienceLabelerModel(**data)
        assert model.quotes == []
    
    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        incomplete_data = {"quotes": ["Sample quote"]}
        
        with pytest.raises(ValidationError) as exc_info:
            JWSTScienceLabelerModel(**incomplete_data)
        
        errors = exc_info.value.errors()
        missing_fields = {error["loc"][0] for error in errors}
        assert "jwstscience" in missing_fields
        assert "reason" in missing_fields
    
    def test_invalid_score_type(self):
        """Test that invalid score types raise ValidationError."""
        invalid_data = {
            "quotes": ["Sample quote"],
            "jwstscience": "invalid",
            "reason": "Sample reason"
        }
        
        with pytest.raises(ValidationError):
            JWSTScienceLabelerModel(**invalid_data)


class TestJWSTDOILabelerModel:
    """Test cases for JWSTDOILabelerModel."""
    
    def test_valid_model_creation(self):
        """Test creating a valid JWSTDOILabelerModel instance."""
        data = {
            "quotes": ["DOI: 10.17909/xyz123", "Data available at MAST"],
            "jwstdoi": 0.8,
            "reason": "Paper mentions specific JWST DOI with 10.17909 prefix"
        }
        model = JWSTDOILabelerModel(**data)
        
        assert model.quotes == data["quotes"]
        assert model.jwstdoi == data["jwstdoi"]
        assert model.reason == data["reason"]
    
    def test_score_validation_range(self):
        """Test that jwstdoi score validation works correctly."""
        valid_data = {
            "quotes": ["Sample quote"],
            "jwstdoi": 0.7,
            "reason": "Sample reason"
        }
        model = JWSTDOILabelerModel(**valid_data)
        assert model.jwstdoi == 0.7
        
        # Edge cases
        edge_cases = [0.0, 1.0]
        for score in edge_cases:
            valid_data["jwstdoi"] = score
            model = JWSTDOILabelerModel(**valid_data)
            assert model.jwstdoi == score
    
    def test_model_serialization(self):
        """Test that models can be serialized to dictionaries."""
        data = {
            "quotes": ["10.17909/example"],
            "jwstdoi": 0.95,
            "reason": "Clear DOI reference found"
        }
        model = JWSTDOILabelerModel(**data)
        serialized = model.model_dump()
        
        assert serialized == data
        assert isinstance(serialized, dict)
    
    def test_model_json_serialization(self):
        """Test that models can be serialized to JSON."""
        data = {
            "quotes": ["10.17909/example"],
            "jwstdoi": 0.95,
            "reason": "Clear DOI reference found"
        }
        model = JWSTDOILabelerModel(**data)
        json_str = model.model_dump_json()
        
        assert isinstance(json_str, str)
        assert "10.17909/example" in json_str
        assert "0.95" in json_str