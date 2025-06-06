"""Report generation functionality."""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from .utils.cache import load_cache

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates analysis reports."""
    
    def __init__(self, 
                 results_dir: Path,
                 science_threshold: float,
                 doi_threshold: float,
                 model_config: Dict[str, str]):
        self.results_dir = results_dir
        self.science_threshold = science_threshold
        self.doi_threshold = doi_threshold
        self.model_config = model_config
        
    def generate_report(self, year_month: str, cache_files: Dict[str, Path]) -> Optional[Dict]:
        """Generate a summary report of the analysis."""
        science_results = load_cache(cache_files['science'])
        doi_results = load_cache(cache_files['doi'])
        skipped_results = load_cache(cache_files['skipped'])
        downloaded_papers = load_cache(cache_files['downloaded'])

        total_attempted = len(downloaded_papers) + len(skipped_results)
        successfully_processed = {
            k: v for k, v in science_results.items() 
            if isinstance(v, dict) and v.get("jwstscience", -1.0) >= 0
        }
        analysis_failed_ids = set(downloaded_papers.keys()) - set(successfully_processed.keys()) - set(skipped_results.keys())
        analysis_failed_count = len(analysis_failed_ids)

        science_papers_ids = {
            arxiv_id for arxiv_id, r in successfully_processed.items()
            if r.get("jwstscience", 0.0) >= self.science_threshold
        }
        science_papers_count = len(science_papers_ids)

        papers_with_dois_count = 0
        papers_missing_dois_count = 0
        detailed_results_list = []

        for arxiv_id in science_papers_ids:
            doi_info = doi_results.get(arxiv_id)
            science_info = successfully_processed[arxiv_id]
            doi_score = 0.0
            doi_reason = "DOI analysis not available or failed"
            doi_quotes: List[str] = []
            has_valid_doi = False 

            if doi_info and isinstance(doi_info, dict) and doi_info.get("jwstdoi", -1.0) >= 0:
                doi_score = doi_info.get("jwstdoi", 0.0)
                doi_reason = doi_info.get("reason", "N/A")
                doi_quotes = doi_info.get("quotes", [])
                if doi_score >= self.doi_threshold:
                    papers_with_dois_count += 1
                    has_valid_doi = True
                else:
                    papers_missing_dois_count += 1
            elif doi_info and isinstance(doi_info, dict) and doi_info.get("jwstdoi", -1.0) < 0:
                doi_reason = doi_info.get("reason", "DOI analysis failed") 
                papers_missing_dois_count += 1
            else: 
                papers_missing_dois_count += 1

            detailed_results_list.append({
                "arxiv_id": arxiv_id,
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
                "science_score": science_info.get("jwstscience"),
                "science_reason": science_info.get("reason"),
                "science_quotes": science_info.get("quotes"),
                "doi_score": doi_score,
                "doi_reason": doi_reason,
                "doi_quotes": doi_quotes,
                "has_valid_doi": has_valid_doi 
            })

        report = {
            "metadata": {
                "report_generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "year_month_analyzed": year_month,
                "science_threshold": self.science_threshold,
                "doi_threshold": self.doi_threshold,
                **self.model_config
            },
            "summary": {
                "total_papers_identified_from_ads": total_attempted,
                "papers_downloaded_and_converted": len(downloaded_papers), 
                "papers_skipped_before_analysis": len(skipped_results),
                "papers_analysis_failed": analysis_failed_count, 
                "papers_successfully_analyzed": len(successfully_processed), 
                "jwst_science_papers_found": science_papers_count,
                "science_papers_with_valid_doi": papers_with_dois_count,
                "science_papers_missing_valid_doi": papers_missing_dois_count,
            },
            "skipped_papers_details": dict(sorted(skipped_results.items())), 
            "jwst_science_papers_details": sorted(detailed_results_list, key=lambda x: x['arxiv_id'])
        }

        report_path = self.results_dir / f"{year_month}_report.json"
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save report file {report_path}: {e}")

        logger.info(f"Report generated: {report_path}")
        
        # Print Summary
        logger.info(f"--- Summary for {year_month} ---")
        logger.info(f"  Total papers from ADS: {report['summary']['total_papers_identified_from_ads']}")
        logger.info(f"  Skipped (Download/Convert): {report['summary']['papers_skipped_before_analysis']}")
        logger.info(f"  Analysis Failed (LLM/etc): {report['summary']['papers_analysis_failed']}")
        logger.info(f"  Successfully Analyzed: {report['summary']['papers_successfully_analyzed']}")
        logger.info(f"  -> JWST Science Papers (Score >= {self.science_threshold}): {science_papers_count}")
        logger.info(f"     -> With Valid DOI (Score >= {self.doi_threshold}): {papers_with_dois_count}")
        logger.info(f"     -> Missing Valid DOI: {papers_missing_dois_count}")
        logger.info("--- End Summary ---")

        return report