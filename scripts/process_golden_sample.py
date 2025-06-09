#!/usr/bin/env python3

import os
import sys
import pandas as pd
import subprocess
from pathlib import Path
from typing import List, Dict
import json
import requests
from dotenv import load_dotenv

def get_arxiv_id_from_bibcode(bibcode: str, api_token: str) -> str | None:
    """
    Converts a NASA ADS Bibcode to an arXiv ID using the ADS API directly.

    Args:
        bibcode: The NASA ADS Bibcode (e.g., '2025apj...980..183w').
        api_token: Your NASA ADS API token.

    Returns:
        The arXiv ID (e.g., '2501.00089') if found, otherwise None.
    """
    api_url = "https://api.adsabs.harvard.edu/v1/search/query"
    headers = {
        "Authorization": f"Bearer {api_token}"
    }
    params = {
        "q": f"bibcode:{bibcode}",
        "fl": "identifier"
    }

    try:
        response = requests.get(api_url, headers=headers, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()

        # The response contains a 'response' dictionary with a 'docs' list
        docs = data.get("response", {}).get("docs", [])

        if not docs:
            return None

        # The first document in the list is our paper
        paper = docs[0]
        identifiers = paper.get("identifier", [])

        for identifier in identifiers:
            if identifier.startswith("arXiv:"):
                return identifier.replace("arXiv:", "")

        return None  # No arXiv ID found in the identifiers

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None

def read_golden_sample(excel_path: str) -> pd.DataFrame:
    """Read the golden sample Excel file."""
    return pd.read_excel(excel_path)

def process_arxiv_id(arxiv_id: str, output_dir: str, reprocess: bool = False) -> Dict:
    """
    Process a single arXiv ID using the command line tool.
    
    Args:
        arxiv_id: The arXiv ID to process
        output_dir: Project directory where papers/, texts/, and results/ subdirectories will be created
        reprocess: If True, force reprocessing even if files exist
    
    Returns:
        Dictionary containing the analysis results
    """
    # Setup paths
    output_path = Path(output_dir)
    papers_dir = output_path / "papers"
    texts_dir = output_path / "texts"
    results_dir = output_path / "results"
    
    # Create directories if they don't exist
    for directory in [papers_dir, texts_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Check for existing files
    pdf_path = papers_dir / f"{arxiv_id}.pdf"
    txt_path = texts_dir / f"{arxiv_id}.txt"
    results_path = results_dir / f"{arxiv_id}_science.json"
    
    # If all files exist and we're not reprocessing, load and return existing results
    if not reprocess and pdf_path.exists() and txt_path.exists() and results_path.exists():
        print(f"Found existing files for {arxiv_id}, loading results...")
        try:
            with open(results_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading existing results for {arxiv_id}: {e}", file=sys.stderr)
            # Continue with reprocessing if results file is corrupted
    
    # Process the paper
    cmd = [
        "jwst-preprint-analyzer",
        "--arxiv-id", arxiv_id,
        "--output-dir", str(output_path),
        "--skip-doi"
    ]
    
    if reprocess:
        cmd.append("--reprocess")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {arxiv_id}: {e.stderr}", file=sys.stderr)
        return {"error": str(e), "arxiv_id": arxiv_id}
    except json.JSONDecodeError as e:
        print(f"Error parsing output for {arxiv_id}: {e}", file=sys.stderr)
        return {"error": "Invalid JSON output", "arxiv_id": arxiv_id}

def main():
    # Load environment variables
    load_dotenv()
    ads_token = os.getenv("ADS_API_KEY")
    if not ads_token:
        print("Error: ADS_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    golden_sample_path = project_root / "golden-sample" / "FlagshipGS-FinalReview.xlsx"
    output_dir = project_root 
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read golden sample
    try:
        df = read_golden_sample(str(golden_sample_path))
    except Exception as e:
        print(f"Error reading golden sample file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Convert bibcodes to arXiv IDs and process papers
    results = []
    for bibcode in df['Bibcode'].dropna().unique():
        print(f"Converting bibcode {bibcode} to arXiv ID...")
        arxiv_id = get_arxiv_id_from_bibcode(bibcode, ads_token)
        
        if arxiv_id:
            print(f"Processing arXiv ID {arxiv_id}...")
            result = process_arxiv_id(arxiv_id, str(output_dir))
            result["bibcode"] = bibcode  # Add bibcode to result for reference
            results.append(result)
        else:
            print(f"No arXiv ID found for bibcode {bibcode}, skipping...")
            results.append({
                "bibcode": bibcode,
                "error": "No arXiv ID found",
                "arxiv_id": None
            })
    
    # Save combined results
    output_file = output_dir / "golden_sample_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessing complete. Results saved to {output_file}")
    print(f"Individual paper results can be found in {output_dir}/results/")

if __name__ == "__main__":
    main() 