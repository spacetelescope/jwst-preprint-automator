import argparse
from pathlib import Path
import logging
import json
import os
import sys
from typing import Optional, List, Dict
import requests
import subprocess
from openai import OpenAI, BadRequestError
import time
from pydantic import BaseModel, Field


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class JWSTScienceLabelerModel(BaseModel):
    quotes: list[str] = Field(..., description="A list of quotes supporting the reason")
    jwstscience: float = Field(..., description="Whether the paper contains JWST science, scored between 0 and 1")
    reason: str = Field(..., description="Justification for the given 'jwstscience' score")

class JWSTDOILabelerModel(BaseModel):
    quotes: list[str] = Field(..., description="A list of quotes supporting the reason")
    jwstdoi: float = Field(..., description="Whether the JWST data is accompanied by a DOI")
    reason: str = Field(..., description="Justification for the given 'jwstdoi' score")

class JWSTPreprintDOIAnalyzer:
    def __init__(self, 
                 year_month: str,  # Format: YYYY-MM
                 output_dir: Path,
                 science_threshold: float = 0.5,
                 doi_threshold: float = 0.8,
                 ads_key: Optional[str] = None,
                 openai_key: Optional[str] = None,
                 reprocess: bool = False,
                 force_llm: bool = False):
        """Initialize the JWST paper analyzer."""
        self.year_month = year_month
        self.reprocess = reprocess
        self.science_threshold = science_threshold
        self.doi_threshold = doi_threshold
        self.force_llm = force_llm
        
        
        # Setup API keys
        self.ads_key = ads_key or os.getenv('ADS_API_KEY')
        self.openai_key = openai_key or os.getenv('OPENAI_API_KEY')
        if not self.ads_key or not self.openai_key:
            raise ValueError("Both ADS_API_KEY and OPENAI_API_KEY must be provided")
        
        # Setup directories
        self.papers_dir = output_dir / "papers"
        self.texts_dir = output_dir / "texts" 
        self.results_dir = output_dir / "results"
        self._setup_directories()
        
        # Setup cache files
        self.download_cache = self.results_dir / f"{self.year_month}_downloaded.json"
        self.science_cache = self.results_dir / f"{self.year_month}_science.json"
        self.doi_cache = self.results_dir / f"{self.year_month}_dois.json"
        self.skipped_cache = self.results_dir / f"{self.year_month}_skipped.json"  # New cache for skipped papers

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.papers_dir, self.texts_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_cache(self, cache_file: Path) -> Dict:
        """Load a cache file if it exists."""
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self, cache_file: Path, data: Dict):
        """Save data to a cache file."""
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _get_paper_list(self) -> List[Dict[str, str]]:
        """Fetch list of astronomy papers for the specified month from ADS."""
        logging.info(f"Fetching paper list for {self.year_month}")
        
        url = "https://api.adsabs.harvard.edu/v1/search/query"
        # params = {
        #     "q": "database:astronomy",
        #     "fq": [
        #         "bibstem:arxiv",
        #         f"pubdate:{self.year_month}"
        #     ],
        #     "fl": ["identifier", "bibcode"],
        #     "rows": 2000,
        #     "sort": "bibcode asc"
        # }
        try:
            year_full, month = self.year_month.split('-')
            year_short = year_full[2:] # Get the last two digits of the year (YY)
            arxiv_pattern = f"arXiv:{year_short}{month}.*"
        else:
            raise ValueError("`year_month` argument format is not YYYY-MM")

        params = {
            "q": "database:astronomy",
            "fq": [
                f'identifier:"{arxiv_pattern}"'
            ],
            "fl": ["identifier", "bibcode"],
            "rows": 2000,
            "sort": "bibcode asc"
        }

        
        headers = {"Authorization": f"Bearer {self.ads_key}"}
        
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        papers = []
        
        for doc in data['response']['docs']:
            arxiv_id = next(
                (id.split(':')[1] for id in doc['identifier'] 
                 if id.startswith('arXiv:')),
                None
            )
            if arxiv_id and 'bibcode' in doc:
                papers.append({
                    'arxiv_id': arxiv_id,
                    'bibcode': doc['bibcode']
                })
        
        logging.info(f"Found {len(papers)} papers")
        return papers

    def _is_downloaded(self, paper: Dict[str, str]) -> bool:
        """Check if paper has already been downloaded and converted."""
        downloaded = self._load_cache(self.download_cache)
        return not self.reprocess and paper['arxiv_id'] in downloaded

    def _is_skipped(self, paper: Dict[str, str]) -> bool:
        """Check if paper was previously skipped."""
        skipped = self._load_cache(self.skipped_cache)
        return not self.reprocess and paper['arxiv_id'] in skipped

    def _mark_as_skipped(self, paper: Dict[str, str], reason: str):
        """Mark a paper as skipped with the given reason."""
        skipped = self._load_cache(self.skipped_cache)
        skipped[paper['arxiv_id']] = {
            "reason": reason,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_cache(self.skipped_cache, skipped)

    def _download_paper(self, paper: Dict[str, str]) -> bool:
        """
        Download paper PDF from arXiv.
        Returns True if successful, False if paper should be skipped.
        """
        logging.info(f"Downloading paper {paper['arxiv_id']}")
        
        if self._is_skipped(paper):
            logging.info(f"Paper {paper['arxiv_id']} was previously skipped")
            return False
        
        pdf_path = self.papers_dir / f"{paper['arxiv_id']}.pdf"
        if pdf_path.exists() and not self.reprocess:
            logging.info(f"Paper {paper['arxiv_id']} already downloaded")
            return True
        
        url = f"https://arxiv.org/pdf/{paper['arxiv_id']}v1"
        headers = {"Authorization": f"Bearer {self.ads_key}"}
        
        try:
            response = requests.get(url, headers=headers, allow_redirects=True)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            # Update download cache
            downloaded = self._load_cache(self.download_cache)
            downloaded[paper['arxiv_id']] = True
            self._save_cache(self.download_cache, downloaded)
            return True
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logging.warning(f"Paper {paper['arxiv_id']} not found (404)")
                self._mark_as_skipped(paper, "404: PDF not found")
            else:
                logging.warning(f"HTTP error downloading {paper['arxiv_id']}: {str(e)}")
                self._mark_as_skipped(paper, f"HTTP error: {str(e)}")
            return False
            
        except Exception as e:
            logging.warning(f"Error downloading {paper['arxiv_id']}: {str(e)}")
            self._mark_as_skipped(paper, f"Download error: {str(e)}")
            return False

    def _convert_to_text(self, paper: Dict[str, str]) -> bool:
        """
        Convert PDF to text using pdftext.
        Returns True if successful, False if conversion failed.
        """
        logging.info(f"Converting paper {paper['arxiv_id']} to text")
        
        pdf_path = self.papers_dir / f"{paper['arxiv_id']}.pdf"
        txt_path = self.texts_dir / f"{paper['arxiv_id']}.txt"
        
        if txt_path.exists() and not self.reprocess:
            logging.info(f"Paper {paper['arxiv_id']} already converted")
            return True
        
        try:
            subprocess.run(
                ["pdftext", "--sort", str(pdf_path), "--out_path", str(txt_path)],
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error converting {paper['arxiv_id']}: {e.stderr}")
            self._mark_as_skipped(paper, f"PDF conversion error: {e.stderr}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error converting {paper['arxiv_id']}: {str(e)}")
            self._mark_as_skipped(paper, f"PDF conversion error: {str(e)}")
            return False

    def _needs_science_analysis(self, paper: Dict[str, str]) -> bool:
        """Check if paper needs JWST science analysis."""
        analyzed = self._load_cache(self.science_cache)
        return self.reprocess or paper['arxiv_id'] not in analyzed

    def _precheck_jwst_mention(self, text: str) -> bool:
        """
        Check if the paper mentions JWST or Webb.
        Returns True if either term is found, False otherwise.
        """
        return "jwst" in text.lower() or "webb" in text.lower()

    def _analyze_science(self, paper: Dict[str, str], validate: bool = True) -> Dict:
        """Analyze paper for JWST science content with optional validation."""
        logging.info(f"Analyzing JWST science content for {paper['arxiv_id']}")
        
        txt_path = self.texts_dir / f"{paper['arxiv_id']}.txt"
        try:
            with open(txt_path, 'r') as f:
                paper_text = f.read()
            
            # Do pre-check unless force_llm is True
            if not self.force_llm and not self._precheck_jwst_mention(paper_text):
                logging.info(f"Paper {paper['arxiv_id']} failed pre-check (no JWST/Webb mention)")
                result = {
                    "jwstscience": 0.0,
                    "quotes": [],
                    "reason": "'Webb' or 'JWST' not found in text (string search)"
                }
                # Cache the result
                analyzed = self._load_cache(self.science_cache)
                analyzed[paper['arxiv_id']] = result
                self._save_cache(self.science_cache, analyzed)
                return result
            
            client = OpenAI(api_key=self.openai_key)
            prompt = f"""
You must determine the likelihood of whether the attached paper text is a James Webb Space Telescope (JWST) science paper. 

Papers that present and/or analyze JWST observations (e.g. using the instruments NIRCam, NIRSpec, NIRISS, MIRI, and FGS) are absolutely science papers. Science can also include comparisons to another paper result based on JWST. Even if the data are distilled into some other JWST science result, and is used as a comparison point in the current paper, then it should be considered a JWST science paper. Note that the JWST science result doesn't have to be the main finding or purpose of the paper, but JWST data must be used to derive some new research. However, mentioning JWST data purely as discussion or motivation does not count. Also, JWST mock data (simulation data) by itself does not mean it's a science paper; JWST science papers must be based on actual JWST observations. Other observatories are not relevant here and should not impact the score.

Include a JSON object with a "jwstscience" score, a "reason" or justification, and a list of "quotes" that directly support the reason. However, since you are an autoregressive LLM, you should cite the quotes first, and then give your science score.

The score for "jwstscience" is a float between 0 and 1, where:

0 - JWST is not mentioned at all at all
0.2 - very low confidence that the paper introduces science results using JWST
0.5 - moderate confidence that the paper introduces JWST science
0.8 - high confidence that the paper introduces new science with JWST
1.0 - absolutely sure that JWST science is presented in the paper

Please only return JSON using an example like one of the following:
{{
    "quotes": ["may be confirmed by follow-up JWST observations of the same systems (Smith et al., in prep)"],
    "jwstscience": 0.2,
    "reason": "motivates future JWST observations that are not presented here"
}}

{{
    "quotes": ["we model using the NIRSpec constraints from Berg et al. (2024) to conclusively rule out shock ionization"],
    "jwstscience": 0.8,
    "reason": "Includes JWST data from another paper to establish a new result that is briefly discussed"
}}


{{
    "quotes": ["structure formation at very high redshifts, e.g., as has recently been shown in recent JWST and ALMA surveys"],
    "jwstscience": 0.1,
    "reason": "is generically motivated by observations but does actually use JWST data"
}}

{{
    "quotes": ["we describe our multi-cycle JWST campaign (program ID 1307)"],
    "jwstscience": 1.0,
    "reason": "New JWST NIRCam observations are presented"
}}

{{
    "quotes": [],
    "jwstscience": 0.0,
    "reason": "does not mention JWST or Webb at all",
}}

Here is the paper converted into plaintext format (please ignore line breaks or malformed tables): 
{paper_text}
"""
            
            result = client.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {
                        "role": "developer", 
                        "content": "You label whether a paper contains JWST science results."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                response_format=JWSTScienceLabelerModel,
            )
            
            initial_result = json.loads(result.choices[0].message.content)
            
            # If validation is requested and we have a non-zero score, validate the result
            if validate and initial_result["jwstscience"] > 0:
                validation_prompt = f"""
You are validating a previous analysis of whether a paper contains JWST science results.
Here are the quotes that were identified:

{json.dumps(initial_result["quotes"], indent=2)}

And here is the reason given:
"{initial_result["reason"]}"

With an assigned score of: {initial_result["jwstscience"]}

Please validate whether this score accurately reflects the JWST science content based on these quotes. You MUST keep the exact same quotes, but you may adjust the score and/or provide a revised reason if you believe the initial assessment was incorrect. 

For example, the original quotes may cite another paper that did JWST science, while the original reason incorrectly stated that new JWST science was being presented. In that case, you should lower the score and change the reasoning to say only that prior JWST scientific results were discussed.

Remember the scoring criteria:
0 - JWST is not mentioned at all
0.2 - very low confidence that the paper introduces science results using JWST
0.5 - moderate confidence that the paper introduces JWST science
0.8 - high confidence that the paper introduces new science with JWST
1.0 - absolutely sure that JWST science is presented in the paper

Please only return JSON using an example like one of the following:
{{
    "quotes": ["may be confirmed by follow-up JWST observations of the same systems (Smith et al., in prep)"],
    "jwstscience": 0.2,
    "reason": "motivates future JWST observations that are not presented here"
}}

{{
    "quotes": ["we model using the NIRSpec constraints from Berg et al. (2024) to conclusively rule out shock ionization"],
    "jwstscience": 0.8,
    "reason": "Includes JWST data from another paper to establish a new result that is briefly discussed"
}}


{{
    "quotes": ["structure formation at very high redshifts, e.g., as has recently been shown in recent JWST and ALMA surveys"],
    "jwstscience": 0.1,
    "reason": "is generically motivated by observations but does actually use JWST data"
}}

{{
    "quotes": ["we describe our multi-cycle JWST campaign (program ID 1307)"],
    "jwstscience": 1.0,
    "reason": "New JWST NIRCam observations are presented"
}}

{{
    "quotes": [],
    "jwstscience": 0.0,
    "reason": "does not mention JWST or Webb at all",
}}

Return a JSON response with the same structure, keeping the quotes identical but potentially
updating the score and/or reason if needed.
"""
                
                validation_result = client.beta.chat.completions.parse(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {
                            "role": "developer",
                            "content": "You validate whether quotes from a paper support the assigned JWST science score."
                        },
                        {
                            "role": "user",
                            "content": validation_prompt
                        }
                    ],
                    response_format=JWSTScienceLabelerModel,
                )
                
                final_result = json.loads(validation_result.choices[0].message.content)
            else:
                final_result = initial_result
            
            # Cache the result
            analyzed = self._load_cache(self.science_cache)
            analyzed[paper['arxiv_id']] = final_result
            self._save_cache(self.science_cache, analyzed)
            
            return final_result
        
        except BadRequestError as e:
            if "maximum context length" in str(e).lower():
                logging.warning(f"Paper {paper['arxiv_id']} exceeds token limit")
                self._mark_as_skipped(paper, "Token limit exceeded: Paper too long for analysis")
            else:
                logging.warning(f"OpenAI API error for {paper['arxiv_id']}: {str(e)}")
                self._mark_as_skipped(paper, f"OpenAI API error: {str(e)}")
            return {"jwstscience": -1.0, "reason": "Analysis failed", "quotes": []}

    def _needs_doi_analysis(self, paper: Dict[str, str]) -> bool:
        """Check if paper needs DOI analysis."""
        analyzed = self._load_cache(self.doi_cache)
        return self.reprocess or paper['arxiv_id'] not in analyzed

    def _analyze_doi(self, paper: Dict[str, str], validate: bool = True) -> Dict:
        """Analyze paper for JWST DOIs with optional validation."""
        logging.info(f"Analyzing JWST DOIs for {paper['arxiv_id']}")
        
        txt_path = self.texts_dir / f"{paper['arxiv_id']}.txt"
        try:
            with open(txt_path, 'r') as f:
                paper_text = f.read()
            
            client = OpenAI(api_key=self.openai_key)
            prompt = f"""
You must determine whether the appended paper contains a JWST DOI. 

It can be assumed that the paper reports scientific results with the James Webb Space Telescope (JWST). We need to scan the paper for a digital object identifier (DOI) that corresponds to the JWST data. The DOI should start with the string "10.17909", otherwise it is not correct. Also, it is possible that other data are associated with DOIs, but the JWST data are not given a DOI; THIS DOES NOT COUNT. Or perhaps the JWST program ID is mentioned, but no DOI is given; again THIS DOES NOT COUNT. Note that the DOI is often found in the Acknowledgments or the Data section.

Include a JSON object with a "jwstdoi" score, a "reason" or justification, and a list of exact "quotes" that directly support the reason. However, since you are an autoregressive LLM, you should cite the quotes first, and then give your jwstdoi score.

The score for "jwstdoi" is a float between 0 and 1, where we give the rough guidelines:

0 - No DOI is provided
0.1 - No DOI number is provided, although a program ID or proposal PI is mentioned
0.5 - A DOI is provided, but it is not clear whether it pertains to the JWST science data or some other data set
1.0 - A DOI beginning with "10.17909/" is explicitly included, or a URL, and the surrounding text explicitly mentions that this DOI is for the JWST data. 

Please only return JSON using an example like one of the following:

{{
    "quotes": [
      "we present new JWST/NIRSpec data as part of the CEERS data release",
      "We acknowledge support from JWST funding",
      ],
    "jwstdoi": 0.1,
    "reason": "Acknowledgments are present, but DOIs are not"
}}

{{
    "quotes": [
      "This work is based on observations made with the NASA/ESA/CSA James Webb Space Telescope. The data were obtained from the Mikulski Archive for Space Telescopes",
      "The specific observations analyzed can be accessed via 10.17909/9bdf-jn24."
      ],
    "jwstdoi": 1.0,
    "reason": "A DOI or link string is given, and the preceding quote confirms that it is for the new JWST data."
}}

Here is the paper converted into plaintext format (please ignore line breaks or malformed tables): 
{paper_text}
"""

            result = client.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {
                        "role": "developer", 
                        "content": "You label whether a paper contains specific DOIs to JWST data."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                response_format=JWSTDOILabelerModel,
            )

            initial_result = json.loads(result.choices[0].message.content)
            
            # If validation is requested and we have a non-zero score, validate the result
            if validate and initial_result["jwstdoi"] > 0:
                validation_prompt = f"""
You are validating a previous analysis of whether a paper properly cites JWST data with DOIs.
Here are the quotes that were identified:

{json.dumps(initial_result["quotes"], indent=2)}

And here is the reason given:
"{initial_result["reason"]}"

With an assigned score of: {initial_result["jwstdoi"]}

Please validate whether this score accurately reflects the JWST DOI citation based on these quotes. You MUST keep the exact same quotes, but you may adjust the score and/or provide a revised reason if you believe the initial assessment was incorrect. 

For example, the original quotes may quote an acknowledgment for JWST funding, but not list a specific DOI, while the original reason incorrectly stated that there was a DOI. In that case, you should lower the jwstdoi score and state that an acknowledgment (but no DOI) was provided.

Remember the scoring criteria:
0 - No DOI is provided
0.1 - No DOI number is provided, although a program ID or proposal PI is mentioned
0.5 - A DOI is provided, but it is not clear whether it pertains to the JWST science data
1.0 - A DOI beginning with "10.17909/" is explicitly included and clearly refers to JWST data

Please only return JSON using an example like one of the following:

{{
    "quotes": [
      "we present new JWST/NIRSpec data as part of the CEERS data release",
      "We acknowledge support from JWST funding",
      ],
    "jwstdoi": 0.1,
    "reason": "Acknowledgments are present, but DOIs are not"
}}

{{
    "quotes": [
      "This work is based on observations made with the NASA/ESA/CSA James Webb Space Telescope. The data were obtained from the Mikulski Archive for Space Telescopes",
      "The specific observations analyzed can be accessed via 10.17909/9bdf-jn24."
      ],
    "jwstdoi": 1.0,
    "reason": "A DOI or link string is given, and the preceding quote confirms that it is for the new JWST data."
}}


Return a JSON response with the same structure, keeping the quotes identical but potentially
updating the score and/or reason if needed.
"""
                
                validation_result = client.beta.chat.completions.parse(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[
                        {
                            "role": "developer",
                            "content": "You validate whether quotes from a paper support the assigned JWST DOI score."
                        },
                        {
                            "role": "user",
                            "content": validation_prompt
                        }
                    ],
                    response_format=JWSTDOILabelerModel,
                )
                
                final_result = json.loads(validation_result.choices[0].message.content)
            else:
                final_result = initial_result
            
            # Cache the result
            analyzed = self._load_cache(self.doi_cache)
            analyzed[paper['arxiv_id']] = final_result
            self._save_cache(self.doi_cache, analyzed)
            
            return final_result
            
        except BadRequestError as e:
            if "maximum context length" in str(e).lower():
                logging.warning(f"Paper {paper['arxiv_id']} exceeds token limit")
                self._mark_as_skipped(paper, "Token limit exceeded: Paper too long for analysis")
            else:
                logging.warning(f"OpenAI API error for {paper['arxiv_id']}: {str(e)}")
                self._mark_as_skipped(paper, f"OpenAI API error: {str(e)}")
            return {"jwstdoi": -1.0, "reason": "Analysis failed", "quotes": []}

    def _generate_report(self):
        """Generate a summary report of the analysis."""
        science_results = self._load_cache(self.science_cache)
        doi_results = self._load_cache(self.doi_cache)
        skipped_results = self._load_cache(self.skipped_cache)
        
        total_papers = len(science_results) + len(skipped_results)
        science_papers = sum(1 for r in science_results.values() 
                           if r["jwstscience"] >= self.science_threshold)
        papers_with_dois = sum(1 for r in doi_results.values() 
                             if r["jwstdoi"] >= self.doi_threshold)
        
        report = {
            "month": self.year_month,
            "thresholds": {
                "jwstscience": self.science_threshold,
                "doi": self.doi_threshold
            },
            "total_papers": total_papers,
            "processed_papers": len(science_results),
            "skipped_papers": len(skipped_results),
            "jwst_science_papers": science_papers,
            "papers_with_dois": papers_with_dois,
            "papers_missing_dois": science_papers - papers_with_dois,
            "skipped_details": skipped_results,
            "detailed_results": {
                arxiv_id: {
                    "science_score": science_results[arxiv_id]["jwstscience"],
                    "doi_score": doi_results.get(arxiv_id, {}).get("jwstdoi", 0),
                    "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}"
                }
                for arxiv_id in science_results
                if science_results[arxiv_id]["jwstscience"] >= self.science_threshold
            }
        }
        
        report_path = self.results_dir / f"{self.year_month}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Report generated: {report_path}")
        return report

    def run(self):
        """Main execution pipeline"""
        try:
            # 1. Get paper list from ADS
            papers = self._get_paper_list()
            
            # 2. Download and convert papers (skip if cached or error)
            processed_papers = []
            for paper in papers:
                if not self._is_downloaded(paper):
                    if self._download_paper(paper):
                        if self._convert_to_text(paper):
                            processed_papers.append(paper)
                        #time.sleep(1)  # Be nice to the arXiv API
                else:
                    processed_papers.append(paper)
            
            # 3. Analyze papers for JWST science
            science_papers = []
            for paper in processed_papers:
                if self._needs_science_analysis(paper):
                    result = self._analyze_science(paper)
                    if result["jwstscience"] >= self.science_threshold:
                        science_papers.append(paper)
            
            # 4. Check DOIs for science papers
            for paper in science_papers:
                if self._needs_doi_analysis(paper):
                    self._analyze_doi(paper)
            
            # 5. Generate summary report
            report = self._generate_report()
            logging.info(f"Analysis complete for {self.year_month}")
            logging.info(f"Total papers processed: {report['processed_papers']}")
            logging.info(f"Papers skipped: {report['skipped_papers']}")
            logging.info(f"JWST science papers found: {report['jwst_science_papers']}")
            logging.info(f"Papers missing DOIs: {report['papers_missing_dois']}")
            
        except Exception as e:
            logging.error(f"Error during analysis: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Analyze arXiv papers for JWST science and DOIs")
    parser.add_argument(
        "year_month",
        help="Month to analyze in YYYY-MM format (e.g., 2024-01)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./"),
        help="Directory to store all outputs"
    )
    parser.add_argument(
        "--science-threshold",
        type=float,
        default=0.5,
        help="Threshold for classifying papers as JWST science (0-1, default: 0.5)"
    )
    parser.add_argument(
        "--doi-threshold",
        type=float,
        default=0.8,
        help="Threshold for considering DOIs as properly cited (0-1, default: 0.8)"
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force reprocessing of already analyzed papers"
    )
    parser.add_argument(
        "--force-llm",
        action="store_true",
        help="Force LLM analysis even if JWST/Webb not found in text"
    )
    parser.add_argument(
        "--ads-key",
        help="ADS API key (will try env var ADS_API_KEY if not provided)"
    )
    parser.add_argument(
        "--openai-key",
        help="OpenAI API key (will try env var OPENAI_API_KEY if not provided)"
    )
    
    args = parser.parse_args()
    
    # Validate thresholds
    if not 0 <= args.science_threshold <= 1:
        parser.error("Science threshold must be between 0 and 1")
    if not 0 <= args.doi_threshold <= 1:
        parser.error("DOI threshold must be between 0 and 1")
    
    try:
        analyzer = JWSTPreprintDOIAnalyzer(
            year_month=args.year_month,
            output_dir=args.output_dir,
            science_threshold=args.science_threshold,
            doi_threshold=args.doi_threshold,
            ads_key=args.ads_key,
            openai_key=args.openai_key,
            reprocess=args.reprocess,
            force_llm=args.force_llm
        )
        analyzer.run()
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
