# JWST Preprint DOI Analyzer

This tool automatically analyzes astronomy preprints from arXiv to identify papers that contain James Webb Space Telescope (JWST) science results and checks whether they properly cite JWST data with Digital Object Identifiers (DOIs) from the Mikulski Archive for Space Telescopes (MAST).

## Purpose

The script performs several key tasks:
1. Downloads astronomy preprints from arXiv via the NASA ADS API for a specified month
2. Converts PDFs to text for analysis
3. Uses string matching to pre-filter papers without JWST mentions
4. Uses gpt-4o-mini to identify papers containing JWST science
5. For papers with JWST science, checks if they include proper JWST data DOIs (beginning with 10.17909)
6. Generates summary reports of the analysis

## Prerequisites

The following environment variables must be set:
- `ADS_API_KEY`: API key for NASA ADS
- `OPENAI_API_KEY`: API key for OpenAI

Required Python packages:
```bash
pip install openai pdftext requests numpy
```

## Usage

Basic usage:
```bash
python jwst-preprint-doi-analyzer.py 2024-01
```

Options:
```bash
python jwst-preprint-doi-analyzer.py [-h] [--output-dir OUTPUT_DIR] [--science-threshold SCIENCE_THRESHOLD]
                                    [--doi-threshold DOI_THRESHOLD] [--reprocess] [--force-llm]
                                    [--ads-key ADS_KEY] [--openai-key OPENAI_KEY] 
                                    month
```

Arguments:
- `month`: Month to analyze in YYYY-MM format (e.g., 2024-01)
- `--output-dir`, `-o`: Directory to store outputs (default: current directory)
- `--science-threshold`: Threshold for classifying papers as JWST science (0-1, default: 0.5)
- `--doi-threshold`: Threshold for considering DOIs as properly cited (0-1, default: 0.8)
- `--reprocess`: Force reprocessing of already analyzed papers
- `--force-llm`: Force LLM analysis even if JWST/Webb not found in text
- `--ads-key`: ADS API key (optional if set in environment)
- `--openai-key`: OpenAI API key (optional if set in environment)

Example with modified thresholds:
```bash
python jwst-preprint-doi-analyzer.py 2024-01 --science-threshold 0.7 --doi-threshold 0.9
```

## Pre-filtering and Error Handling

The script includes several optimizations and error handling features:

1. **JWST Mention Pre-check**: By default, papers are first checked for mentions of "JWST" or "Webb" before running more expensive LLM analysis. Use `--force-llm` to bypass this check.

2. **Error Handling**:
   - Papers not found on arXiv (404 errors)
   - PDFs that fail to download or convert
   - Papers that exceed the LLM token limit
   - Failed API calls

All errors are gracefully handled and tracked in the results.

## Directory Structure

The script creates and maintains the following directory structure:
```
.
├── papers/           # Downloaded PDF files
│   └── *.pdf
├── texts/           # Converted text files
│   └── *.txt
└── results/         # Analysis results
    ├── YYYY-MM_downloaded.json  # Cache of downloaded papers
    ├── YYYY-MM_science.json     # JWST science analysis results
    ├── YYYY-MM_dois.json        # DOI analysis results
    ├── YYYY-MM_skipped.json     # Tracking of skipped papers
    └── YYYY-MM_report.json      # Summary report
```

## Output Files

### Science Analysis (YYYY-MM_science.json)
```json
{
  "2401.00934": {
    "science": 1.0,
    "reason": "Presents new JWST observations and analyzes the size evolution of quiescent galaxies using JWST/NIRCam data.",
    "quotes": [
      "we use new ultra-deep JWST/NIRCam imaging from the JWST Advanced Deep Extragalactic Survey (JADES)",
      "With deep multi-band NIRCam images in GOODS-South from JADES"
    ]
  },
  "2401.00990": {
    "science": 0.0,
    "reason": "'Webb' or 'JWST' not found in text (string search)",
    "quotes": []
  },
  "2401.00999": {
    "science": -1.0,
    "reason": "Analysis failed",
    "quotes": []
  }
}
```

### Skipped Papers (YYYY-MM_skipped.json)
```json
{
  "2401.00123": {
    "reason": "404: PDF not found",
    "timestamp": "2024-01-30 14:35:08"
  },
  "2401.00456": {
    "reason": "Token limit exceeded: Paper too long for analysis",
    "timestamp": "2024-01-30 14:36:22"
  }
}
```

### DOI Analysis (YYYY-MM_dois.json)
```json
{
  "2401.00934": {
    "jwstdoi": 0.2,
    "reason": "The paper discusses JWST data and methodologies extensively but does not explicitly mention a DOI associated with JWST data.",
    "quotes": [
      "The arrival of Cycle 1 JWST data demonstrated the power of this new facility to identify and characterize robust massive quiescent galaxy candidates at z > 2.5",
      "we use new ultra-deep JWST/NIRCam imaging from the JWST Advanced Deep Extragalactic Survey (JADES, Eisenstein et al. 2023) to investigate the redshift evolution of the half-light sizes of quiescent galaxies."
    ]
  }
}
```

### Summary Report (YYYY-MM_report.json)
```json
{
  "month": "2024-01",
  "total_papers": 10,
  "processed_papers": 8,
  "skipped_papers": 2,
  "jwst_science_papers": 1,
  "papers_with_dois": 0,
  "papers_missing_dois": 1,
  "skipped_details": {
    "2401.00123": {
      "reason": "404: PDF not found",
      "timestamp": "2024-01-30 14:35:08"
    }
  },
  "detailed_results": {
    "2401.00934": {
      "science_score": 1.0,
      "doi_score": 0.5,
      "arxiv_url": "https://arxiv.org/abs/2401.00934"
    }
  }
}
```

## Scoring System

### Science Score
- -1.0: Analysis failed (e.g., paper too long)
- 0.0: JWST not mentioned or not a science paper
- 0.2: Very low confidence of JWST science
- 0.5: Moderate confidence of JWST science
- 0.8: High confidence of JWST science
- 1.0: Definite JWST science

### DOI Score
- 0.0: No DOI provided
- 0.5: DOI mentioned but unclear if for JWST data
- 1.0: Clear JWST data DOI present

Papers are considered JWST science papers if they score ≥ 0.5 on the science scale, and are flagged as properly citing JWST data if they score ≥ 0.8 on the DOI scale.

## Caching

The script caches results at each stage to avoid reprocessing papers unnecessarily. Use the `--reprocess` flag to force reanalysis of previously processed papers.