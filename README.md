# JWST Preprint Automation

This script classifies whether papers are JWST science papers. Optionally, it can also check for proper JWST DOI citations from MAST. The tool uses GPT-4.1-nano for improved text snippet reranking to identify the most relevant content before final classification.

## Quick start
There are two main ways to use this JWST Preprint Automation package:

**Batch mode**: `jwst-preprint-analyzer --year-month YYYY-MM`. For example:

```bash
jwst-preprint-analyzer --year-month 2025-05
```

**Individual mode**: `jwst-preprint-analyzer --arxiv-id YYMM.NNNNN`. For example:

```bash
jwst-preprint-analyzer --arxiv-id 2309.16028
```

## Installation

We recommend using version Python 3.10 or higher, and using a virtual environment. This has so far only been tested on macOS and Linux.

To install from the source, first copy the repository to your computer
```bash
git clone git@github.com:spacetelescope/jwst-preprint-automator.git

cd jwst-preprint-automator
```

Then, create a virtual environment. An easy way to do this is using `uv`:
```bash
uv venv && source .venv/bin/activate
uv sync
```

Alternatively, you could install with python's built in venv and pip:
```bash
python3 -m venv .venv
source .venv/bin/activate 
# on Windows, instead do 
# .venv\Scripts\activate

pip install -e . # install in editable mode
```

### Environment Variables
Next, you must set a few API keys, which enable the use of LLMs and querying ADS. Create a `.env` file in the project root with the following contents:
```bash
export OPENAI_API_KEY=your_openai_key_here  # Required for all classification use cases
export COHERE_API_KEY=your_cohere_key_here  # Optional: for legacy reranking (GPT reranker used by default)
export ADS_API_KEY=your_ads_key_here        # Required for batch mode
```


## Additional usage patterns

The are many more options that you can view using `jwst-preprint-analyzer -h`. Here is a brief menu of possibilities:

```bash
# do not try to classify DOIs
jwst-preprint-analyzer --arxiv-id 2501.00089 --skip-doi

# specify the LLM
jwst-preprint-analyzer --arxiv-id 2503.18791 --gpt-model gpt-4.1-mini-2025-04-14

# reprocess the script and save in different directory
jwst-preprint-analyzer --arxiv-id 2503.18791 --reprocess --output-dir ./results-reprocessed

# use the legacy Cohere reranker instead of GPT
jwst-preprint-analyzer --arxiv-id 2503.18791 --no-gpt-reranker
```

**Detail options:**
-   `-h, --help`: Show this help message and exit.
-   `--output-dir OUTPUT_DIR, -o OUTPUT_DIR`: Project directory wherein the following subdirectories will be created:
    - `papers/`: Downloaded PDF files
    - `texts/`: Extracted text from PDFs
    - `results/`: Analysis results and cache files
    Default: current directory (`./`).
-   `--prompts-dir PROMPTS_DIR, -p PROMPTS_DIR`: Directory containing LLM prompt template files (e.g., `science_system.txt`). Default: `./prompts`.
-   `--science-threshold SCIENCE_THRESHOLD`: Threshold for classifying papers as JWST science (0-1). Default: `0.5`.
-   `--doi-threshold DOI_THRESHOLD`: Threshold for considering DOIs as properly cited (0-1). Default: `0.8`.
-   `--reranker-threshold RERANKER_THRESHOLD`: Minimum reranker score for the top snippet to proceed with LLM analysis. Scores below this will skip LLM. Default: `0.05`.
-   `--reprocess`: Force reprocessing of downloaded/analyzed papers, ignoring caches.
-   `--top-k-snippets TOP_K_SNIPPETS`: Number of top reranked snippets to send to the LLM. Default: `5`.
-   `--context-sentences CONTEXT_SENTENCES`: Number of sentences before and after a keyword sentence to include in a snippet. Default: `3`.
-   `--reranker-model RERANKER_MODEL`: Cohere reranker model name (when using legacy reranking). Default: `rerank-v3.5`.
-   `--gpt-model GPT_MODEL`: OpenAI GPT model for science and DOI analysis. Default: `gpt-4o-mini-2024-07-18`.
-   `--validate-llm`: Perform a second LLM call to validate the first analysis (increases cost/time).
-   `--skip-doi`: Skip DOI analysis completely, even for papers that meet the science threshold.
-   `--no-gpt-reranker`: Use the legacy Cohere reranker instead of the default GPT-4.1-nano reranker.

**API keys:**
-   `--ads-key ADS_KEY`: ADS API key (optional if `ADS_API_KEY` environment variable is set).
-   `--openai-key OPENAI_KEY`: OpenAI API key (optional if `OPENAI_API_KEY` environment variable is set).
-   `--cohere-key COHERE_KEY`: Cohere API key (optional if `COHERE_API_KEY` environment variable is set; only needed for legacy reranking).


## Outputs

### Batch Mode
Generates a summary report at `results/YYYY-MM_report.json` with:
- Paper counts and analysis results
- Details on JWST science papers found
- DOI citation compliance status

### Single Paper Mode  
Outputs JSON to stdout with science/DOI analysis results for the specified arXiv paper.

