"""Command-line interface for JWST Preprint DOI Analyzer."""

import argparse
import logging
import re
import sys
from pathlib import Path

from .analyzer import JWSTPreprintDOIAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Download arXiv papers via ADS, extract text, rerank snippets, and use LLMs to classify JWST science content and DOI presence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--lookback-days",
        type=int,
        help="Number of days to look back for recent papers (default: 1 for batch processing)."
    )
    mode_group.add_argument(
        "--arxiv-id", 
        help="Specific arXiv ID (e.g., 2301.12345) to process for single paper analysis."
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./"),
        help="Project directory where papers/, texts/, and results/ subdirectories will be created. Defaults to current directory."
    )
    parser.add_argument(
        "--prompts-dir", "-p",
        type=Path,
        default=Path("./prompts"), 
        help="Directory containing LLM prompt template files (e.g., science_system.txt)."
    )
    parser.add_argument(
        "--science-threshold",
        type=float,
        default=0.5, 
        help="Threshold for classifying papers as JWST science (0-1)"
    )
    parser.add_argument(
        "--doi-threshold",
        type=float,
        default=0.8, 
        help="Threshold for considering DOIs as properly cited (0-1)"
    )
    parser.add_argument(
        "--reranker-threshold",
        type=float,
        default=0.05,
        help="Minimum reranker score for the top snippet to proceed with LLM analysis. Scores below this threshold will skip the LLM call (range 0-1)."
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force reprocessing of downloaded/analyzed papers"
    )
    parser.add_argument(
        "--top-k-snippets",
        type=int,
        default=5, 
        help="Number of top reranked snippets to send to the LLM"
    )
    parser.add_argument(
        "--context-sentences",
        type=int,
        default=3, 
        help="Number of sentences before and after a keyword sentence to include in a snippet"
    )
    parser.add_argument(
        "--cohere-reranker-model",
        default="rerank-v3.5", 
        help="Cohere reranker model name (when using legacy reranking)"
    )
    parser.add_argument(
        "--gpt-model",
        default="gpt-4.1-mini-2025-04-14", 
        help="GPT scoring model for JWST science and DOIs"
    )
    parser.add_argument(
        "--validate-llm",
        action="store_true",
        help="Perform a second LLM call to validate the first analysis (increases cost/time)"
    )
    parser.add_argument(
        "--skip-doi",
        action="store_true",
        help="Skip DOI analysis completely, even for papers that meet the science threshold"
    )
    parser.add_argument(
        "--no-gpt-reranker",
        action="store_true",
        help="Use the legacy Cohere reranker instead of the default GPT-4.1-nano reranker"
    )
    parser.add_argument(
        "--limit-papers",
        type=int,
        help="Limit processing to the first N papers (useful for testing). Only applies to batch mode."
    )
    parser.add_argument("--ads-key", help="ADS API key (uses ADS_API_KEY env var if not provided)")
    parser.add_argument("--openai-key", help="OpenAI API key (uses OPENAI_API_KEY env var if not provided)")
    parser.add_argument("--cohere-key", help="Cohere API key (uses COHERE_API_KEY env var if not provided; reranking skipped if missing)")

    args = parser.parse_args()

    # Validate thresholds
    if not 0 <= args.science_threshold <= 1:
        parser.error("Science threshold must be between 0 and 1")
    if not 0 <= args.doi_threshold <= 1:
        parser.error("DOI threshold must be between 0 and 1")
    
    
    # Validate limit-papers
    if args.limit_papers is not None and args.limit_papers < 1:
        parser.error("--limit-papers must be a positive integer")
    
    # Validate lookback-days
    if args.lookback_days is not None and args.lookback_days < 1:
        parser.error("--lookback-days must be a positive integer")
    
    if args.arxiv_id and not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", args.arxiv_id.split('/')[-1]):
        parser.error("Invalid arXiv ID format. Should be like XXXX.YYYYY or XXXX.YYYYYvN")

    # Create output/prompts directory if it doesn't exist
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.prompts_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create necessary directories ({args.output_dir}, {args.prompts_dir}): {e}")
        sys.exit(1)

    try:
        analyzer = JWSTPreprintDOIAnalyzer(
            lookback_days=args.lookback_days,
            arxiv_id=args.arxiv_id,
            output_dir=args.output_dir,
            prompts_dir=args.prompts_dir, 
            science_threshold=args.science_threshold,
            doi_threshold=args.doi_threshold,
            reranker_threshold=args.reranker_threshold,
            ads_key=args.ads_key,
            openai_key=args.openai_key,
            cohere_key=args.cohere_key,
            gpt_model=args.gpt_model,
            cohere_reranker_model=args.cohere_reranker_model,
            top_k_snippets=args.top_k_snippets,
            context_sentences=args.context_sentences,
            validate_llm=args.validate_llm,
            reprocess=args.reprocess,
            skip_doi=args.skip_doi,
            use_gpt_reranker=not args.no_gpt_reranker,
            limit_papers=args.limit_papers,
        )

        if analyzer.run_mode == "batch":
            analyzer.run_batch() 
        elif analyzer.run_mode == "single":
            analyzer.process_single_paper(args.arxiv_id) 

    except ValueError as e:
        logger.error(f"Initialization Error: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"Setup Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during execution: {e}")
        logger.exception("Traceback:")
        sys.exit(1)

    logger.info("Script finished successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
