"""ADS API client for fetching paper metadata."""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class ADSClient:
    """Client for interacting with NASA ADS API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.adsabs.harvard.edu/v1/search/query"
    
    def _calculate_date_range(self, lookback_days: int) -> Dict[str, str]:
        """Calculate date range for ADS query based on lookback days.
        
        Args:
            lookback_days: Number of days to look back from today
            
        Returns:
            Dictionary with formatted date strings for the query
        """
        today = datetime.now()
        start_date = today - timedelta(days=lookback_days)
        
        return {
            'entdate_start': start_date.strftime('%Y-%m-%d'),
            'entdate_end': today.strftime('%Y-%m-%d'),
            'year_start': today.year - 1,
            'year_end': today.year,
            'fulltext_mtime_start': '1000-01-01T00:00:00.000Z'
        }
    
    def get_recent_papers(self, lookback_days: int = 1) -> List[Dict[str, any]]:
        """Fetch list of JWST-related arXiv papers from ADS using pagination.
        
        Args:
            lookback_days: Number of days to look back from today (default: 1)
            
        Returns:
            List of paper dictionaries with all ADS fields
        """
        logger.info(f"Fetching JWST papers from last {lookback_days} day(s) with row-based pagination")
        
        # Calculate date ranges
        dates = self._calculate_date_range(lookback_days)
        
        # Build the query string
        query_parts = [
            'full:("JWST" OR "Webb Space Telescope" OR "Webb Telescope")',
            f'entdate:[{dates["entdate_start"]} TO {dates["entdate_end"]}]',
            f'year:[{dates["year_start"]} TO {dates["year_end"]}]',
            'bibstem:(arXiv)',
            f'fulltext_mtime:["{dates["fulltext_mtime_start"]}" TO *]'
        ]
        query = ' '.join(query_parts)
        
        # Field list from the user's example
        fields = [
            'id', 'title', 'abstract', 'keyword', 'bibcode', 'bibstem', 'doi',
            'author', 'pubdate', 'entdate', 'entry_date', 'first_author', 'pub',
            'volume', 'page', 'citation_count', 'property', 'orcid_pub',
            'orcid_user', 'orcid_other', 'aff', 'issue', 'identifier',
            'fulltext_mtime', 'alternate_bibcode'
        ]
        
        all_papers = []
        start = 0
        rows_per_page = 20
        
        while True:
            params = {
                'q': query,
                'fl': ','.join(fields),
                'rows': rows_per_page,
                'start': start
            }
            headers = {'Authorization': f'Bearer {self.api_key}'}
            
            logger.info(f"Querying ADS with start={start}, rows={rows_per_page}")
            
            try:
                response = requests.get(self.base_url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
            except requests.exceptions.Timeout:
                logger.error(f"ADS API request timed out at start={start}")
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"ADS API request failed at start={start}: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"ADS Response Status: {e.response.status_code}")
                    logger.error(f"ADS Response Body: {e.response.text}")
                    if e.response.status_code == 401:
                        logger.error("Check if your ADS_API_KEY is valid.")
                break
            
            try:
                data = response.json()
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON response from ADS at start={start}: {response.text[:500]}")
                break
            
            if 'response' not in data or 'docs' not in data['response']:
                logger.warning(f"Unexpected ADS response format at start={start}")
                break
            
            docs = data['response']['docs']
            num_found = data['response'].get('numFound', 0)
            logger.info(f"Retrieved {len(docs)} papers (total available: {num_found})")
            
            if not docs:
                # No more results
                break
            
            # Process each document
            for doc in docs:
                # Extract arXiv ID from identifiers
                identifiers = doc.get('identifier', [])
                if not isinstance(identifiers, list):
                    identifiers = [identifiers]
                
                arxiv_id = None
                for id_str in identifiers:
                    if isinstance(id_str, str) and id_str.startswith('arXiv:'):
                        arxiv_id = id_str.split(':', 1)[1]
                        break
                
                # Validate arXiv ID format
                if arxiv_id and re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", arxiv_id.split('/')[-1]):
                    # Store all fields from ADS
                    paper = {
                        'arxiv_id': arxiv_id,
                        'arxiv_url': f"https://arxiv.org/abs/{arxiv_id}",
                        **doc  # Include all ADS fields
                    }
                    all_papers.append(paper)
                elif arxiv_id:
                    logger.warning(f"Extracted arXiv ID '{arxiv_id}' has unexpected format. Skipping.")
                else:
                    logger.debug(f"No valid arXiv ID found in identifiers: {identifiers}")
            
            # Check if we've retrieved all results
            if len(docs) < rows_per_page:
                # Last page
                break
            
            # Move to next page
            start += rows_per_page
        
        logger.info(f"Found total of {len(all_papers)} JWST papers")
        return all_papers
