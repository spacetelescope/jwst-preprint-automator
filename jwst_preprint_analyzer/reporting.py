"""Report generation functionality."""

import csv
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
        
    def generate_report(self, batch_identifier: str, cache_files: Dict[str, Path], limit_papers: Optional[int] = None) -> Optional[Dict]:
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

        metadata = {
            "report_generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "batch_identifier": batch_identifier,
            "science_threshold": self.science_threshold,
            "doi_threshold": self.doi_threshold,
            **self.model_config
        }
        
        if limit_papers is not None:
            metadata["limit_papers"] = limit_papers
        
        report = {
            "metadata": metadata,
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

        json_filename = f"{batch_identifier}_report"
        if limit_papers is not None:
            json_filename += f"_limit{limit_papers}"
        json_filename += ".json"
        report_path = self.results_dir / json_filename
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save report file {report_path}: {e}")

        logger.info(f"Report generated: {report_path}")
        
        # Print Summary
        logger.info(f"--- Summary for {batch_identifier} ---")
        logger.info(f"  Total papers from ADS: {report['summary']['total_papers_identified_from_ads']}")
        logger.info(f"  Skipped (Download/Convert): {report['summary']['papers_skipped_before_analysis']}")
        logger.info(f"  Analysis Failed (LLM/etc): {report['summary']['papers_analysis_failed']}")
        logger.info(f"  Successfully Analyzed: {report['summary']['papers_successfully_analyzed']}")
        logger.info(f"  -> JWST Science Papers (Score >= {self.science_threshold}): {science_papers_count}")
        logger.info(f"     -> With Valid DOI (Score >= {self.doi_threshold}): {papers_with_dois_count}")
        logger.info(f"     -> Missing Valid DOI: {papers_missing_dois_count}")
        logger.info("--- End Summary ---")

        return report

    def generate_csv_report(self, batch_identifier: str, cache_files: Dict[str, Path], limit_papers: Optional[int] = None) -> Optional[Path]:
        """Generate a CSV report with ALL papers and their processing status, including all ADS fields."""
        science_results = load_cache(cache_files['science'])
        doi_results = load_cache(cache_files['doi'])
        skipped_results = load_cache(cache_files['skipped'])
        downloaded_papers = load_cache(cache_files['downloaded'])
        papers_cache = load_cache(cache_files['papers'])
        
        # Create CSV data for ALL papers in the papers cache
        csv_data = []
        
        for arxiv_id, paper_info in papers_cache.items():
            # Helper function to format list fields
            def format_field(value):
                if isinstance(value, list):
                    return "|".join(str(v) for v in value)
                return value if value is not None else ""
            
            # Start with ALL ADS fields from paper_info
            row = {
                "arxiv_id": arxiv_id,
                "arxiv_url": paper_info.get("arxiv_url", f"https://arxiv.org/abs/{arxiv_id}"),
                "paper_title": format_field(paper_info.get("title", "")),
                "bibcode": paper_info.get("bibcode", ""),
                "entry_date": paper_info.get("entry_date", ""),
                "pubdate": paper_info.get("pubdate", ""),
                "abstract": format_field(paper_info.get("abstract", "")),
                "keyword": format_field(paper_info.get("keyword", "")),
                "doi": format_field(paper_info.get("doi", "")),
                "author": format_field(paper_info.get("author", "")),
                "first_author": paper_info.get("first_author", ""),
                "pub": paper_info.get("pub", ""),
                "volume": paper_info.get("volume", ""),
                "page": format_field(paper_info.get("page", "")),
                "citation_count": paper_info.get("citation_count", ""),
                "property": format_field(paper_info.get("property", "")),
                "orcid_pub": format_field(paper_info.get("orcid_pub", "")),
                "orcid_user": format_field(paper_info.get("orcid_user", "")),
                "orcid_other": format_field(paper_info.get("orcid_other", "")),
                "aff": format_field(paper_info.get("aff", "")),
                "issue": paper_info.get("issue", ""),
                "identifier": format_field(paper_info.get("identifier", "")),
                "fulltext_mtime": paper_info.get("fulltext_mtime", ""),
                "alternate_bibcode": format_field(paper_info.get("alternate_bibcode", "")),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "top_quotes": "",
                "jwst_sciencescore": 0.0,
                "jwst_sciencereason": "",
                "jwst_doiscore": 0.0,
                "jwst_doireason": "",
                "jwst_classification": ""
            }
            
            # Check if paper was skipped before analysis
            if arxiv_id in skipped_results:
                skip_info = skipped_results[arxiv_id]
                row.update({
                    "jwst_classification": skip_info.get("reason", ""),
                    "timestamp": skip_info.get("timestamp", ""),
                    "jwst_sciencereason": "Skipped before analysis"
                })
            
            # Check if paper was downloaded but analysis failed
            elif arxiv_id in downloaded_papers and arxiv_id not in science_results:
                row.update({
                    "jwst_classification": "Analysis failed - no science result",
                    "jwst_sciencereason": "Analysis failed"
                })
            
            # Check if paper had science analysis
            elif arxiv_id in science_results:
                science_info = science_results[arxiv_id]

                if isinstance(science_info, dict):
                    # Handle quotes formatting
                    quotes = science_info.get("quotes", [])
                    if isinstance(quotes, list):
                        quotes_str = "|".join(quotes)
                    else:
                        quotes_str = str(quotes)

                    row.update({
                        "jwst_sciencescore": science_info.get("jwstscience", 0.0),
                        "jwst_sciencereason": science_info.get("reason", ""),
                        "top_quotes": quotes_str
                    })

                    # Determine status based on science analysis
                    jwst_score = science_info.get("jwstscience", 0.0)
                    science_reason = science_info.get("reason", "")

                    if jwst_score < 0:
                        row["jwst_classification"] = "Science analysis failed"
                    elif "No relevant keywords" in science_reason:
                        row["jwst_classification"] = "No JWST keywords found"
                    elif "reranker score" in science_reason:
                        row["jwst_classification"] = "Below reranker threshold"
                    elif jwst_score < self.science_threshold:
                        row["jwst_classification"] = "Below science threshold"
                    else:
                        row["jwst_classification"] = "JWST science paper"

                    # Add DOI information if available
                    if arxiv_id in doi_results:
                        doi_info = doi_results[arxiv_id]
                        if isinstance(doi_info, dict):
                            row["jwst_doiscore"] = doi_info.get("jwstdoi", 0.0)
                            row["jwst_doireason"] = doi_info.get("reason", "")
                else:
                    row.update({
                        "jwst_classification": "Invalid science analysis result",
                        "jwst_sciencereason": "Analysis failed"
                    })
            
            # Paper wasn't processed at all
            else:
                row.update({
                    "jwst_classification": "Not processed",
                    "jwst_sciencereason": "Paper not processed"
                })
            
            csv_data.append(row)
        
        # Sort by arxiv_id
        csv_data.sort(key=lambda x: x['arxiv_id'])
        
        # Write CSV file
        csv_filename = f"{batch_identifier}_report"
        if limit_papers is not None:
            csv_filename += f"_limit{limit_papers}"
        csv_filename += ".csv"
        csv_path = self.results_dir / csv_filename
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                # Define all fieldnames including ADS fields
                fieldnames = [
                    "arxiv_id", "arxiv_url", "paper_title", "bibcode", "entry_date", "pubdate",
                    "abstract", "keyword", "doi", "author", "first_author", "pub",
                    "volume", "page", "citation_count", "property", "orcid_pub",
                    "orcid_user", "orcid_other", "aff", "issue", "identifier",
                    "fulltext_mtime", "alternate_bibcode",
                    "top_quotes", "jwst_sciencescore", "jwst_sciencereason",
                    "jwst_doiscore", "jwst_doireason", "timestamp", "jwst_classification"
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC, doublequote=False, escapechar='\\')
                writer.writeheader()
                writer.writerows(csv_data)
                
            logger.info(f"CSV report generated: {csv_path}")
            return csv_path
            
        except Exception as e:
            logger.error(f"Failed to save CSV report file {csv_path}: {e}")
            return None