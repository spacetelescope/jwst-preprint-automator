"""Tests for ADS client functionality."""

import pytest
from datetime import datetime, timedelta
from jwst_preprint_analyzer.clients.ads import ADSClient


class TestADSClient:
    """Test suite for ADS client."""
    
    def test_calculate_date_range_one_day(self):
        """Test date range calculation for 1-day lookback."""
        client = ADSClient(api_key="dummy_key")
        dates = client._calculate_date_range(lookback_days=1)
        
        # Verify structure
        assert 'entdate_start' in dates
        assert 'entdate_end' in dates
        assert 'year_start' in dates
        assert 'year_end' in dates
        assert 'fulltext_mtime_start' in dates
        
        # Verify date formats
        assert len(dates['entdate_start']) == 10  # YYYY-MM-DD
        assert len(dates['entdate_end']) == 10
        assert dates['fulltext_mtime_start'] == '1000-01-01T00:00:00.000Z'
        
        # Verify year range
        current_year = datetime.now().year
        assert dates['year_start'] == current_year - 1
        assert dates['year_end'] == current_year
        
        # Verify date difference is 1 day
        start_date = datetime.strptime(dates['entdate_start'], '%Y-%m-%d')
        end_date = datetime.strptime(dates['entdate_end'], '%Y-%m-%d')
        assert (end_date - start_date).days == 1
    
    def test_calculate_date_range_seven_days(self):
        """Test date range calculation for 7-day lookback."""
        client = ADSClient(api_key="dummy_key")
        dates = client._calculate_date_range(lookback_days=7)
        
        # Verify date difference is 7 days
        start_date = datetime.strptime(dates['entdate_start'], '%Y-%m-%d')
        end_date = datetime.strptime(dates['entdate_end'], '%Y-%m-%d')
        assert (end_date - start_date).days == 7
    
    def test_calculate_date_range_today(self):
        """Test that end date is today."""
        client = ADSClient(api_key="dummy_key")
        dates = client._calculate_date_range(lookback_days=1)
        
        today = datetime.now().strftime('%Y-%m-%d')
        assert dates['entdate_end'] == today
