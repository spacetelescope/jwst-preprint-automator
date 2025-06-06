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

def process_arxiv_id(arxiv_id: str, output_dir: str) -> Dict:
    """Process a single arXiv ID using the command line tool."""
    # Create the output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "jwst-preprint-analyzer",
        "--arxiv-id", arxiv_id,
        "--output-dir", str(output_path),
        "--doi-threshold", "0"  # Set DOI threshold to 0 to skip DOI validation
    ]
    
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
    output_dir = project_root / "results_test-golden-sample"
    
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