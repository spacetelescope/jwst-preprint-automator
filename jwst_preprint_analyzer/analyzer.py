"""Main analyzer class that orchestrates the analysis pipeline."""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, List

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .clients.ads import ADSClient
from .clients.openai import OpenAIClient
from .clients.cohere import CohereClient
from .clients.gpt_reranker import GPTReranker
from .processing.downloader import PaperDownloader
from .processing.converter import PDFConverter
from .processing.text_extractor import TextExtractor
from .analysis.science import ScienceAnalyzer
from .analysis.doi import DOIAnalyzer
from .reporting import ReportGenerator
from .utils.cache import load_cache, save_cache
from .utils.prompts import load_prompts

logger = logging.getLogger(__name__)


class JWSTPreprintDOIAnalyzer:
    """Main analyzer class for JWST preprint DOI analysis."""
    
    def __init__(self,
                 output_dir: Path,
                 lookback_days: Optional[int] = None,
                 arxiv_id: Optional[str] = None, 
                 prompts_dir: Path = Path("./prompts"),
                 science_threshold: float = 0.5,
                 doi_threshold: float = 0.8,
                 reranker_threshold: float = 0.1,
                 ads_key: Optional[str] = None,
                 openai_key: Optional[str] = None,
                 cohere_key: Optional[str] = None,
                 gpt_model: str = 'gpt-4.1-mini-2025-04-14',
                 cohere_reranker_model: str = 'rerank-v3.5', 
                 top_k_snippets: int = 15,
                 context_sentences: int = 3,
                 validate_llm: bool = False,
                 reprocess: bool = False,
                 skip_doi: bool = False,
                 use_gpt_reranker: bool = True,
                 limit_papers: Optional[int] = None):
        """Initialize the JWST paper analyzer."""
        
        # Validate that exactly one mode is provided
        modes_provided = sum([bool(lookback_days is not None), bool(arxiv_id)])
        if modes_provided != 1:
            raise ValueError("Exactly one of 'lookback_days' or 'arxiv_id' must be provided.")

        self.lookback_days = lookback_days
        self.single_arxiv_id = arxiv_id
        self.run_mode = "batch" if lookback_days is not None else "single"
        
        self.reprocess = reprocess
        self.science_threshold = science_threshold
        self.doi_threshold = doi_threshold
        self.reranker_threshold = reranker_threshold
        self.gpt_model = gpt_model
        self.cohere_reranker_model = cohere_reranker_model 
        self.top_k_snippets = top_k_snippets
        self.context_sentences = context_sentences
        self.validate_llm = validate_llm
        self.skip_doi = skip_doi
        self.use_gpt_reranker = use_gpt_reranker
        self.limit_papers = limit_papers

        # Setup API keys
        self.ads_key = ads_key or os.getenv('ADS_API_KEY')
        self.openai_key = openai_key or os.getenv('OPENAI_API_KEY')
        self.cohere_key = cohere_key or os.getenv('COHERE_API_KEY')
        
        if self.run_mode == "batch" and not self.ads_key:
            logger.warning("ADS_API_KEY not provided. Batch mode ('lookback_days') will fail.")
        if not self.openai_key:
            raise ValueError("OPENAI_API_KEY must be provided (as argument or environment variable)")

        # Initialize clients
        self.ads_client = ADSClient(self.ads_key) if self.ads_key else None
        self.openai_client = OpenAIClient(self.openai_key, self.gpt_model)
        self.cohere_client = CohereClient(self.cohere_key, self.cohere_reranker_model)
        self.gpt_reranker = GPTReranker(self.openai_client, 'gpt-4.1-nano-2025-04-14') if self.use_gpt_reranker else None
        
        # Create directories
        self.output_dir = output_dir 
        self.papers_dir = output_dir / "papers"
        self.texts_dir = output_dir / "texts"
        self.results_dir = output_dir / "results"
        self.prompts_dir = prompts_dir
        self._setup_directories()
        
        # Initialize components
        self.downloader = PaperDownloader(self.papers_dir)
        self.converter = PDFConverter(self.texts_dir)
        self.text_extractor = TextExtractor(self.context_sentences)
        
        # Load prompts
        self.prompts = load_prompts(self.prompts_dir)
        
        # Initialize analyzers
        self.science_analyzer = ScienceAnalyzer(
            self.openai_client, self.cohere_client, self.text_extractor,
            self.prompts, self.top_k_snippets, self.reranker_threshold, self.validate_llm,
            self.gpt_reranker
        )
        self.doi_analyzer = DOIAnalyzer(
            self.openai_client, self.cohere_client, self.text_extractor,
            self.prompts, self.top_k_snippets, self.validate_llm,
            self.gpt_reranker
        )
        
        # Setup cache files
        if self.lookback_days is not None:
            # Use date-based prefix for batch mode
            from datetime import datetime
            cache_prefix = datetime.now().strftime('%Y-%m-%d')
        else:
            cache_prefix = self.single_arxiv_id if self.single_arxiv_id else "single_run"
            
        self.cache_files = {
            'downloaded': self.results_dir / f"{cache_prefix}_downloaded.json",
            'science': self.results_dir / f"{cache_prefix}_science.json",
            'doi': self.results_dir / f"{cache_prefix}_dois.json",
            'skipped': self.results_dir / f"{cache_prefix}_skipped.json",
            'snippets': self.results_dir / f"{cache_prefix}_snippets.json",
            'papers': self.results_dir / f"{cache_prefix}_papers.json"
        }
        
        # Initialize report generator
        model_config = {
            "gpt_model": self.gpt_model,
            "reranker_type": "GPT-4.1-nano-2025-04-14" if self.use_gpt_reranker else "Cohere",
            "cohere_reranker_model": self.cohere_reranker_model if self.cohere_client.client else "N/A (Cohere unavailable)",
            "top_k_snippets": self.top_k_snippets,
            "context_sentences": self.context_sentences,
            "llm_validation_enabled": self.validate_llm,
            "prompts_directory": str(self.prompts_dir.resolve()),
        }
        self.report_generator = ReportGenerator(
            self.results_dir, self.science_threshold, self.doi_threshold, model_config
        )

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.papers_dir, self.texts_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _is_downloaded_and_converted(self, paper: Dict[str, str]) -> bool:
        """Check if paper PDF is downloaded AND text file exists."""
        downloaded_cache = load_cache(self.cache_files['downloaded'])
        txt_path = self.texts_dir / f"{paper['arxiv_id']}.txt"
        pdf_path = self.papers_dir / f"{paper['arxiv_id']}.pdf"

        # Check if files exist and are not empty
        files_exist = pdf_path.exists() and txt_path.exists() and txt_path.stat().st_size > 0

        # In single paper mode, we only need to check if files exist
        if self.run_mode == "single":
            return not self.reprocess and files_exist

        # In batch mode, we also check the cache
        return (not self.reprocess
                and paper['arxiv_id'] in downloaded_cache
                and files_exist)

    def _is_skipped(self, paper: Dict[str, str]) -> bool:
        """Check if paper was previously skipped."""
        if self.run_mode == "single":
            return False

        skipped = load_cache(self.cache_files['skipped'])
        return not self.reprocess and paper['arxiv_id'] in skipped

    def _mark_as_skipped(self, paper: Dict[str, str], reason: str, save_to_cache: bool = True):
        """Mark a paper as skipped with the given reason."""
        arxiv_id = paper['arxiv_id']
        logger.warning(f"Skipping paper {arxiv_id}: {reason}")

        if save_to_cache and self.run_mode == "batch":
            skipped = load_cache(self.cache_files['skipped'])
            skipped[arxiv_id] = {
                "reason": reason,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_cache(self.cache_files['skipped'], skipped)

            downloaded = load_cache(self.cache_files['downloaded'])
            if arxiv_id in downloaded:
                del downloaded[arxiv_id]
                save_cache(self.cache_files['downloaded'], downloaded)

    def _needs_analysis(self, paper: Dict[str, str], cache_key: str) -> bool:
        """Generic check if paper needs analysis based on cache and reprocess flag."""
        if self.run_mode == 'single':
            return True
        analyzed = load_cache(self.cache_files[cache_key])
        return self.reprocess or paper['arxiv_id'] not in analyzed

    def _process_paper(self, paper: Dict[str, str]) -> bool:
        """Download and convert a paper. Returns True if successful."""
        arxiv_id = paper['arxiv_id']
        
        # Download
        if not self.downloader.download_paper(arxiv_id, self.reprocess):
            self._mark_as_skipped(paper, "Download failed", save_to_cache=True)
            return False
            
        # Convert
        pdf_path = self.papers_dir / f"{arxiv_id}.pdf"
        if not self.converter.convert_to_text(arxiv_id, pdf_path, self.reprocess):
            self._mark_as_skipped(paper, "PDF conversion failed", save_to_cache=True)
            return False
            
        # Update download cache in both modes
        downloaded = load_cache(self.cache_files['downloaded'])
        downloaded[arxiv_id] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "success"
        }
        save_cache(self.cache_files['downloaded'], downloaded)
            
        return True

    def run_batch(self):
        """Main execution pipeline for batch mode."""
        if self.run_mode != "batch":
            logger.error("run_batch called in non-batch mode.")
            return

        start_time = time.time()
        from datetime import datetime
        batch_identifier = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"Starting analysis for {batch_identifier} (lookback: {self.lookback_days} days)...")

        try:
            # Get paper list from ADS with pagination
            papers = self.ads_client.get_recent_papers(self.lookback_days)
            if not papers:
                logger.warning("No papers found or ADS query failed. Exiting.")
                self.report_generator.generate_report(batch_identifier, self.cache_files, self.limit_papers)
                return
                
            # Apply limit if specified
            if self.limit_papers is not None:
                original_count = len(papers)
                papers = papers[:self.limit_papers]
                logger.info(f"Limiting processing to first {len(papers)} papers out of {original_count} total papers")
            
            # Store paper metadata in cache (with ALL ADS fields)
            papers_cache = load_cache(self.cache_files['papers'])
            for paper in papers:
                arxiv_id = paper['arxiv_id']
                if arxiv_id not in papers_cache:
                    # Store all ADS fields for this paper
                    papers_cache[arxiv_id] = paper
            save_cache(self.cache_files['papers'], papers_cache)

            # Process each paper
            for i, paper in enumerate(papers):
                arxiv_id = paper['arxiv_id']
                logger.info(f"Processing paper {i+1}/{len(papers)}: {arxiv_id}")

                if self._is_skipped(paper):
                    logger.info(f"Paper {arxiv_id} was previously skipped. Skipping.")
                    continue

                # Download and convert
                if not self._is_downloaded_and_converted(paper):
                    if not self._process_paper(paper):
                        continue

                # Analyze for JWST Science
                txt_path = self.texts_dir / f"{arxiv_id}.txt"
                science_result = None
                
                if self._needs_analysis(paper, 'science'):
                    science_result = self.science_analyzer.analyze(arxiv_id, txt_path)
                    if science_result and "error" not in science_result:
                        science_cache = load_cache(self.cache_files['science'])
                        science_cache[arxiv_id] = science_result
                        save_cache(self.cache_files['science'], science_cache)
                else:
                    science_cache = load_cache(self.cache_files['science'])
                    science_result = science_cache.get(arxiv_id)
                    if science_result and isinstance(science_result, dict):
                        logger.info(f"Using cached science analysis for {arxiv_id}")
                    else:
                        logger.warning(f"Cache logic error: Science cache missing or invalid for {arxiv_id}. Re-analyzing.")
                        science_result = self.science_analyzer.analyze(arxiv_id, txt_path)
                        if science_result and "error" not in science_result:
                            science_cache[arxiv_id] = science_result
                            save_cache(self.cache_files['science'], science_cache)

                # Check science score
                current_science_score = -1.0
                if science_result and isinstance(science_result, dict) and "error" not in science_result:
                    current_science_score = science_result.get("jwstscience", -1.0)
                else:
                    logger.warning(f"Science analysis failed for {arxiv_id}, cannot proceed to DOI analysis.")
                    continue

                if current_science_score < self.science_threshold:
                    logger.info(f"Paper {arxiv_id} does not meet science threshold ({current_science_score:.2f} < {self.science_threshold}). Skipping DOI analysis.")
                    continue

                # Skip DOI analysis if flag is set
                if self.skip_doi:
                    logger.info(f"DOI analysis skipped for {arxiv_id} due to --skip-doi flag")
                    continue

                # Analyze for DOIs
                if self._needs_analysis(paper, 'doi'):
                    doi_result = self.doi_analyzer.analyze(arxiv_id, txt_path)
                    if doi_result and doi_result.get("jwstdoi", -1.0) >= 0:
                        doi_cache = load_cache(self.cache_files['doi'])
                        doi_cache[arxiv_id] = doi_result
                        save_cache(self.cache_files['doi'], doi_cache)
                    else:
                        logger.warning(f"DOI analysis failed for science paper {arxiv_id}.")
                else:
                    doi_cache = load_cache(self.cache_files['doi'])
                    doi_result_cached = doi_cache.get(arxiv_id)
                    if doi_result_cached and isinstance(doi_result_cached, dict):
                        logger.info(f"Using cached DOI analysis for {arxiv_id}")
                    else:
                        logger.warning(f"DOI analysis needed for {arxiv_id} but cache missing/incomplete. Re-analyzing.")
                        doi_result = self.doi_analyzer.analyze(arxiv_id, txt_path)
                        if doi_result and doi_result.get("jwstdoi", -1.0) >= 0:
                            doi_cache[arxiv_id] = doi_result
                            save_cache(self.cache_files['doi'], doi_cache)

            # Generate final summary report
            self.report_generator.generate_report(batch_identifier, self.cache_files, self.limit_papers)
            
            # Generate CSV report
            self.report_generator.generate_csv_report(batch_identifier, self.cache_files, self.limit_papers)

            end_time = time.time()
            logger.info(f"Analysis complete for {batch_identifier} in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            logger.exception(f"An unhandled error occurred during the run: {e}")
            try:
                logger.info("Attempting to generate partial report after error...")
                self.report_generator.generate_report(batch_identifier, self.cache_files)
            except Exception as report_err:
                logger.error(f"Failed to generate partial report: {report_err}")
            raise

    def _transform_analysis_keys(self, analysis_result: Dict) -> Dict:
        """Transform internal analysis keys to user-facing keys."""
        if not analysis_result or not isinstance(analysis_result, dict):
            return analysis_result

        transformed = {}
        key_mapping = {
            'jwstscience': 'jwst_sciencescore',
            'jwstdoi': 'jwst_doiscore',
            'reason': 'jwst_sciencereason' if 'jwstscience' in analysis_result else 'jwst_doireason'
        }

        for key, value in analysis_result.items():
            new_key = key_mapping.get(key, key)
            transformed[new_key] = value

        return transformed

    def process_single_paper(self, arxiv_id: str):
        """Processes a single paper by arXiv ID and prints results to stdout."""
        start_time = time.time()
        logger.info(f"Starting SINGLE analysis for arXiv ID: {arxiv_id}")

        # Basic validation
        if not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", arxiv_id.split('/')[-1]):
            logger.error(f"Invalid arXiv ID format provided: {arxiv_id}")
            print(json.dumps({"error": "invalid_arxiv_id", "arxiv_id": arxiv_id,
                            "message": "Invalid arXiv ID format."}, indent=2))
            return

        # Create a dummy paper dictionary
        paper = {'arxiv_id': arxiv_id, 'bibcode': 'N/A'}

        final_output = {
            "arxiv_id": arxiv_id,
            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
            "processed_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "jwst_classification": "Started",
            "science_analysis": None,
            "doi_analysis": None,
            "error_info": None
        }

        try:
            # Download and convert
            if not self._process_paper(paper):
                logger.error(f"Failed to download/convert {arxiv_id}.")
                final_output["jwst_classification"] = "Error: Download/Conversion Failed"
                final_output["error_info"] = "Could not retrieve or convert PDF from arXiv."
                print(json.dumps(final_output, indent=2, ensure_ascii=False))
                return

            # Analyze science
            txt_path = self.texts_dir / f"{arxiv_id}.txt"
            logger.info(f"Analyzing science content for single paper {arxiv_id}")
            science_result = self.science_analyzer.analyze(arxiv_id, txt_path)
            final_output["science_analysis"] = self._transform_analysis_keys(science_result)

            current_science_score = -1.0
            if science_result and isinstance(science_result, dict) and "error" not in science_result:
                current_science_score = science_result.get("jwstscience", -1.0)
                final_output["jwst_classification"] = "Science Analysis Complete"
            else:
                logger.error(f"Science analysis failed for {arxiv_id}. See results.")
                final_output["jwst_classification"] = "Error: Science Analysis Failed"
                transformed_result = self._transform_analysis_keys(science_result) if science_result else {}
                final_output["error_info"] = transformed_result.get("jwst_sciencereason", "Science analysis failed")
                print(json.dumps(final_output, indent=2, ensure_ascii=False))
                return

            # Analyze DOI if science score meets threshold and not skipped
            if current_science_score >= self.science_threshold and not self.skip_doi:
                logger.info(f"Science score >= threshold. Analyzing DOI for {arxiv_id}")
                doi_result = self.doi_analyzer.analyze(arxiv_id, txt_path)
                final_output["doi_analysis"] = self._transform_analysis_keys(doi_result)

                if doi_result and isinstance(doi_result, dict) and "error" not in doi_result:
                    final_output["jwst_classification"] = "Complete"
                else:
                    logger.error(f"DOI analysis failed for {arxiv_id}. See results.")
                    final_output["jwst_classification"] = "Complete (with DOI Analysis Error)"
            else:
                skip_reason = "Skipped due to --skip-doi flag" if self.skip_doi else "Skipped due to low science score"
                logger.info(f"DOI analysis skipped for {arxiv_id}: {skip_reason}")
                final_output["jwst_classification"] = f"Complete (DOI Skipped - {skip_reason})"
                final_output["doi_analysis"] = {"jwst_doiscore": 0.0, "jwst_doireason": skip_reason,
                                               "quotes": [], "error": None}

        except Exception as e:
            logger.exception(f"Unhandled error during single paper processing for {arxiv_id}: {e}")
            final_output["jwst_classification"] = "Error: Unhandled Exception"
            final_output["error_info"] = f"Unexpected error: {str(e)}"

        try:
            print(json.dumps(final_output, indent=2, ensure_ascii=False))
        except Exception as json_e:
            logger.error(f"Failed to serialize final result to JSON for {arxiv_id}: {json_e}")
            print(f'{{"error": "json_serialization_failed", "arxiv_id": "{arxiv_id}", "message": "{str(json_e)}"}}')