"""ADS API client for fetching paper metadata."""

import json
import logging
import re
from typing import List, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class ADSClient:
    """Client for interacting with NASA ADS API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.adsabs.harvard.edu/v1/search/query"
        
    def get_papers_for_month(self, year_month: str) -> List[Dict[str, str]]:
        """Fetch list of astronomy papers for the specified month from ADS."""
        logger.info(f"Fetching paper list for {year_month}")

        try:
            year_full, month = year_month.split('-')
            if len(year_full) != 4 or not year_full.isdigit() or len(month) != 2 or not month.isdigit():
                raise ValueError("Invalid format")
            year_short = year_full[2:]  # Get the last two digits of the year (YY)
            arxiv_pattern = f"arXiv:{year_short}{month}.*"
        except ValueError:
            logger.error("`year_month` argument format must be YYYY-MM")
            raise ValueError("`year_month` argument format must be YYYY-MM")

        params = {
            "q": "database:astronomy",
            "fq": [f'identifier:"{arxiv_pattern}"'],
            "fl": ["identifier", "bibcode"],
            "rows": 2000,
            "sort": "bibcode asc"
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            response = requests.get(self.base_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logger.error("ADS API request timed out.")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"ADS API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"ADS Response Status: {e.response.status_code}")
                logger.error(f"ADS Response Body: {e.response.text}")
                if e.response.status_code == 401:
                    logger.error("Check if your ADS_API_KEY is valid.")
            return []

        try:
            data = response.json()
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON response from ADS: {response.text[:500]}")
            return []

        papers = []
        if 'response' not in data or 'docs' not in data['response']:
            logger.warning(f"Unexpected ADS response format for {year_month}")
            return []

        for doc in data['response']['docs']:
            identifiers = doc.get('identifier', [])
            if not isinstance(identifiers, list):
                identifiers = [identifiers]

            arxiv_id = next(
                (id_str.split(':')[1] for id_str in identifiers
                 if isinstance(id_str, str) and id_str.startswith('arXiv:')),
                None
            )
            if arxiv_id and re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", arxiv_id.split('/')[-1]):
                if 'bibcode' in doc:
                    papers.append({
                        'arxiv_id': arxiv_id,
                        'bibcode': doc['bibcode']
                    })
            elif arxiv_id:
                logger.warning(f"Extracted potential arXiv ID '{arxiv_id}' has unexpected format. Skipping.")

        logger.info(f"Found {len(papers)} valid papers")
        return papers