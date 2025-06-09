"""PDF download functionality."""

import logging
from pathlib import Path
from typing import Dict

import requests
from ..utils.cache import load_cache, save_cache

logger = logging.getLogger(__name__)


class PaperDownloader:
    """Handles downloading papers from arXiv."""
    
    def __init__(self, papers_dir: Path):
        self.papers_dir = papers_dir
        
    def download_paper(self, arxiv_id: str, reprocess: bool = False) -> bool:
        """
        Download paper PDF from arXiv.
        Returns True if successful, False if paper should be skipped.
        """
        logger.info(f"Checking download status for paper {arxiv_id}")
        
        pdf_path = self.papers_dir / f"{arxiv_id}.pdf"
        
        # Check if paper is already downloaded and not reprocessing
        if pdf_path.exists() and not reprocess:
            logger.info(f"Paper {arxiv_id} already downloaded.")
            return True

        logger.info(f"Downloading paper {arxiv_id}...")
        url = f"https://arxiv.org/pdf/{arxiv_id}v1"
        headers = {} 

        try:
            response = requests.get(url, headers=headers, allow_redirects=True)
            response.raise_for_status()

            with open(pdf_path, 'wb') as f:
                f.write(response.content)
                
            return True

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Paper {arxiv_id} not found (404)")
                return False
            else:
                logger.warning(f"HTTP error downloading {arxiv_id}: {str(e)}")
                return False

        except Exception as e:
            logger.warning(f"Error downloading {arxiv_id}: {str(e)}")
            return False