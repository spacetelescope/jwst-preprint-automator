# JWST Preprint DOI Analyzer

This tool automatically analyzes astronomy preprints from arXiv to identify papers that contain James Webb Space Telescope (JWST) science results and checks whether they properly cite JWST data with Digital Object Identifiers (DOIs) from the Mikulski Archive for Space Telescopes (MAST). It can operate in a batch mode for a given month or analyze a single preprint by its arXiv ID.

## Purpose

The script performs several tasks:
1.  **Batch Mode:** Downloads astronomy preprints from arXiv via the NASA ADS API for a specified month (`--year-month`).
2.  **Single Paper Mode:** Downloads a specific preprint from arXiv given its ID (`--arxiv-id`).
3.  Converts downloaded PDFs to text using the `pdftext` command-line tool.
4.  Extracts relevant text snippets based on predefined keywords.
5.  Optionally reranks these snippets for relevance using the Cohere API.
6.  Uses OpenAI GPT models (e.g., `gpt-4.1-mini-2025-04-14`) to:
    * Identify papers containing JWST science.
    * For papers with JWST science, check if they include proper JWST data DOIs (specifically those with the prefix `10.17909`).
7.  Optionally performs a second LLM call to validate the initial analysis.
8.  Generates a detailed JSON summary report for batch processing or outputs JSON results to stdout for single paper analysis.
9.  Manages LLM interaction through customizable prompt templates stored in a `prompts/` directory.

## Prerequisites

### Environment Variables
The following environment variables should be set:
-   `ADS_API_KEY`: API key for NASA ADS (required for batch mode).
-   `OPENAI_API_KEY`: API key for OpenAI (required).
-   `COHERE_API_KEY`: API key for Cohere (optional; if not provided, snippet reranking will be skipped).

### External Tools
-   **`pdftext`**: This command-line tool must be installed and accessible in your system's PATH. It is used for converting PDF files to plain text.

### Python Packages
Required Python packages:
```bash
pip install openai requests pydantic nltk cohere
```
(Note: `nltk` may also require you to download the 'punkt' tokenizer: `python -c "import nltk; nltk.download('punkt')"`)

## Usage

The script can be run in two modes: batch processing for a whole month or analysis of a single paper.

### Batch Mode (by Month)
Analyzes all astronomy preprints for a given month and year.
```bash
python jwst-preprint-doi-analyzer.py --year-month YYYY-MM
```
Example:
```bash
python jwst-preprint-doi-analyzer.py --year-month 2024-01 --output-dir ./analysis_results --prompts-dir ./custom_prompts
```

### Single Paper Mode (by arXiv ID)
Analyzes a single preprint specified by its arXiv ID.
```bash
python jwst-preprint-doi-analyzer.py --arxiv-id XXXXX.YYYYY
```
Example:
```bash
python jwst-preprint-doi-analyzer.py --arxiv-id 2301.12345 --gpt-model gpt-4o
```

### Command-Line Options
```bash
python jwst-preprint-doi-analyzer.py [-h] (--year-month YEAR_MONTH | --arxiv-id ARXIV_ID)
                                    [--output-dir OUTPUT_DIR] [--prompts-dir PROMPTS_DIR]
                                    [--science-threshold SCIENCE_THRESHOLD]
                                    [--doi-threshold DOI_THRESHOLD]
                                    [--reranker-threshold RERANKER_THRESHOLD]
                                    [--reprocess] [--top-k-snippets TOP_K_SNIPPETS]
                                    [--context-sentences CONTEXT_SENTENCES]
                                    [--reranker-model RERANKER_MODEL]
                                    [--gpt-model GPT_MODEL] [--validate-llm]
                                    [--ads-key ADS_KEY] [--openai-key OPENAI_KEY]
                                    [--cohere-key COHERE_KEY]
```

**Mode Selection (choose one):**
-   `--year-month YEAR_MONTH`: Month to analyze in `YYYY-MM` format (e.g., `2024-01`) for batch processing.
-   `--arxiv-id ARXIV_ID`: Specific arXiv ID (e.g., `2301.12345`) to process for single paper analysis.

**General Options:**
-   `-h, --help`: Show this help message and exit.
-   `--output-dir OUTPUT_DIR, -o OUTPUT_DIR`: Directory to store all outputs (papers, texts, results). Default: current directory (`./`).
-   `--prompts-dir PROMPTS_DIR, -p PROMPTS_DIR`: Directory containing LLM prompt template files (e.g., `science_system.txt`). Default: `./prompts`.
-   `--science-threshold SCIENCE_THRESHOLD`: Threshold for classifying papers as JWST science (0-1). Default: `0.5`.
-   `--doi-threshold DOI_THRESHOLD`: Threshold for considering DOIs as properly cited (0-1). Default: `0.8`.
-   `--reranker-threshold RERANKER_THRESHOLD`: Minimum reranker score for the top snippet to proceed with LLM analysis. Scores below this will skip LLM. Default: `0.05`.
-   `--reprocess`: Force reprocessing of downloaded/analyzed papers, ignoring caches.
-   `--top-k-snippets TOP_K_SNIPPETS`: Number of top reranked snippets to send to the LLM. Default: `5`.
-   `--context-sentences CONTEXT_SENTENCES`: Number of sentences before and after a keyword sentence to include in a snippet. Default: `3`.
-   `--reranker-model RERANKER_MODEL`: Cohere reranker model name. Default: `rerank-v3.5`.
-   `--gpt-model GPT_MODEL`: OpenAI GPT model for science and DOI analysis. Default: `gpt-4.1-mini-2025-04-14`.
-   `--validate-llm`: Perform a second LLM call to validate the first analysis (increases cost/time).

**API Key Options:**
-   `--ads-key ADS_KEY`: ADS API key (optional if `ADS_API_KEY` environment variable is set).
-   `--openai-key OPENAI_KEY`: OpenAI API key (optional if `OPENAI_API_KEY` environment variable is set).
-   `--cohere-key COHERE_KEY`: Cohere API key (optional if `COHERE_API_KEY` environment variable is set; reranking skipped if missing).


## Analysis Workflow & Error Handling

1.  **Paper Acquisition**: Downloads PDF from arXiv (either via ADS for batch or directly for single ID).
2.  **Text Conversion**: Converts PDF to text using `pdftext`.
3.  **Snippet Extraction**: Identifies sentences containing specified keywords (related to JWST, instruments, data archives) and expands them with surrounding context.
4.  **Snippet Reranking (Optional)**: If a Cohere API key is provided, snippets are reranked based on relevance to specific queries (for science and DOI analysis separately).
5.  **LLM Analysis**:
    * If the reranked snippets (or original snippets if reranking is skipped) meet the `--reranker-threshold`, they are sent to the specified OpenAI GPT model.
    * The LLM classifies the paper for JWST science content and, if applicable, for the presence of MAST DOIs, based on the provided snippets and prompt templates.
    * An optional validation step can perform a second LLM call to verify the initial findings.
6.  **Error Handling**:
    * Gracefully handles common issues such as:
        * Papers not found on arXiv (404 errors).
        * PDFs that fail to download or convert.
        * Empty or problematic text conversions.
        * Failures in API calls (ADS, OpenAI, Cohere).
        * Papers exceeding token limits for LLM analysis (though snippet extraction aims to mitigate this).
    * Skipped or failed papers are logged and reported.

## Prompt Management

The script uses text files for LLM prompts, loaded from the directory specified by `--prompts-dir` (default: `./prompts`). This allows for easy customization of the instructions given to the LLM for both science and DOI analysis, as well as their validation and reranking queries.

Expected prompt files include:
-   `science_system.txt`, `science_user.txt`
-   `science_validate_system.txt`, `science_validate_user.txt` (if `--validate-llm` is used)
-   `rerank_science_query.txt`
-   `doi_system.txt`, `doi_user.txt`
-   `doi_validate_system.txt`, `doi_validate_user.txt` (if `--validate-llm` is used)
-   `rerank_doi_query.txt`

If essential prompt files are missing, the script will raise an error.

## Directory Structure

The script creates and maintains the following directory structure within the specified `--output-dir`:
```
<output_dir>/
├── papers/                 
│   └── <arxiv_id>.pdf
├── texts/                  
│   └── <arxiv_id>.txt
├── results/                
│   ├── <YYYY-MM>_downloaded.json 
│   ├── <YYYY-MM>_science.json    
│   ├── <YYYY-MM>_dois.json       
│   ├── <YYYY-MM>_skipped.json    
│   ├── <YYYY-MM>_snippets.json   
│   └── <YYYY-MM>_report.json     
└── prompts/                
    └── *.txt
```
For single paper mode (`--arxiv-id`), output is directed to `stdout`, though `papers/` and `texts/` directories under `--output-dir` might still be used for intermediate files if not already cached and `--reprocess` isn't used. Cache files in `results/` are prefixed with the arXiv ID in single mode if caching were implemented for it (currently, single mode does not save to these monthly caches but uses `--reprocess` for re-download/conversion).

## Output Files and Format

### Batch Mode (`--year-month`)

The primary output for batch mode is the `YYYY-MM_report.json` file in the `results/` directory. Other JSON files in `results/` store intermediate data and caches.

**Summary Report (`YYYY-MM_report.json`) Example:**
```json
{
  "metadata": {
    "report_generated": "2024-05-07 18:30:00 UTC",
    "year_month_analyzed": "2024-01",
    "science_threshold": 0.5,
    "doi_threshold": 0.8,
    "gpt_model": "gpt-4.1-mini-2025-04-14",
    "reranker_model": "rerank-v3.5",
    "top_k_snippets": 5,
    "context_sentences": 3,
    "llm_validation_enabled": false,
    "prompts_directory": "/path/to/prompts"
  },
  "summary": {
    "total_papers_identified_from_ads": 500,
    "papers_downloaded_and_converted": 490,
    "papers_skipped_before_analysis": 10,
    "papers_analysis_failed": 5,
    "papers_successfully_analyzed": 485,
    "jwst_science_papers_found": 50,
    "science_papers_with_valid_doi": 20,
    "science_papers_missing_valid_doi": 30
  },
  "skipped_papers_details": {
    "2401.00123": {
      "reason": "404: PDF not found",
      "timestamp": "2024-01-30 14:35:08"
    }
    // ... more skipped papers
  },
  "jwst_science_papers_details": [
    {
      "arxiv_id": "2401.00934",
      "arxiv_url": "[https://arxiv.org/abs/2401.00934](https://arxiv.org/abs/2401.00934)",
      "science_score": 1.0,
      "science_reason": "Presents new JWST observations...",
      "science_quotes": ["quote1", "quote2"],
      "doi_score": 0.2,
      "doi_reason": "Does not explicitly mention a DOI...",
      "doi_quotes": ["quote_doi1"],
      "has_valid_doi": false
    }
    // ... more science papers
  ]
}
```

**Other cache files (examples):**

* **`YYYY-MM_science.json` / `YYYY-MM_dois.json`:**
    ```json
    {
      "2401.00934": {
        "jwstscience": 1.0, // or "jwstdoi"
        "reason": "Justification for the score...",
        "quotes": ["Supporting quote 1", "Supporting quote 2"]
      }
      // ... more papers
    }
    ```
* **`YYYY-MM_snippets.json`:** Stores the reranked snippets that were sent to the LLM for each paper and analysis type (science/DOI).
* **`YYYY-MM_skipped.json`:** Details papers skipped due to download, conversion, or critical analysis errors.

### Single Paper Mode (`--arxiv-id`)

Output is a JSON object printed to standard output (`stdout`).
**Example Output:**
```json
{
  "arxiv_id": "2301.12345",
  "arxiv_url": "[https://arxiv.org/abs/2301.12345](https://arxiv.org/abs/2301.12345)",
  "processed_timestamp": "2024-05-07 19:00:00 UTC",
  "status": "Complete",
  "science_analysis": {
    "quotes": ["Detailed analysis of NIRCam images...", "The JWST program ID G01234..."],
    "jwstscience": 1.0,
    "reason": "The paper clearly presents results from new JWST observations using NIRCam."
  },
  "doi_analysis": {
    "quotes": ["Data used in this study are available at MAST archive under DOI 10.17909/abcdef."],
    "jwstdoi": 1.0,
    "reason": "The paper explicitly provides a MAST DOI (10.17909) for the JWST data."
  },
  "error_info": null
}
```
If an error occurs, `status` will reflect it, and `error_info` may contain details. If DOI analysis is skipped due to a low science score, `doi_analysis` will indicate this.

## Scoring System

### JWST Science Score (`jwstscience`)
-   `-1.0`: Analysis failed (e.g., text file missing, LLM error).
-   `0.0`: No relevant keywords found, or LLM determines no JWST science content / reranker score below threshold.
-   `(0.0, 1.0]`: A float representing the LLM's assessed likelihood of the paper containing JWST science.
    -   Values closer to `1.0` indicate higher confidence.
    -   The `--science-threshold` (default `0.5`) is used to determine if a paper is classified as "JWST science."

### JWST DOI Score (`jwstdoi`)
-   `-1.0`: Analysis failed.
-   `0.0`: No relevant keywords found, or LLM determines no relevant DOI / paper not identified as JWST science.
-   `(0.0, 1.0]`: A float representing the LLM's assessed likelihood of a proper JWST data DOI (prefix `10.17909`) being present and correctly cited.
    -   Values closer to `1.0` indicate higher confidence.
    -   The `--doi-threshold` (default `0.8`) is used to determine if a paper is classified as "having a valid DOI."

## Caching

In batch mode, the script caches results at various stages to avoid reprocessing papers unnecessarily:
-   Download status (`YYYY-MM_downloaded.json`)
-   Science analysis results (`YYYY-MM_science.json`)
-   DOI analysis results (`YYYY-MM_dois.json`)
-   Skipped paper records (`YYYY-MM_skipped.json`)
-   Extracted and reranked snippets (`YYYY-MM_snippets.json`)

Use the `--reprocess` flag to ignore these caches and force re-analysis of all papers in the batch.
Single paper mode currently does not write to these monthly cache files but will reuse downloaded PDFs and converted text files from `papers/` and `texts/` subdirectories if they exist, unless `--reprocess` is specified.
```