import argparse
from pathlib import Path
import logging
import json
import os
import sys
import re
import time
from typing import Optional, List, Dict, Tuple, Set, Union
import requests
import subprocess
from openai import OpenAI, BadRequestError
from pydantic import BaseModel, Field
import nltk 
import cohere 
from nltk.tokenize import sent_tokenize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class JWSTScienceLabelerModel(BaseModel):
    quotes: list[str] = Field(..., description="A list of quotes supporting the reason, MUST be exact substrings from the provided excerpts.")
    jwstscience: float = Field(..., description="Whether the paper contains JWST science, scored between 0 and 1")
    reason: str = Field(..., description="Justification for the given 'jwstscience' score based ONLY on the provided excerpts")

class JWSTDOILabelerModel(BaseModel):
    quotes: list[str] = Field(..., description="A list of quotes supporting the reason, MUST be exact substrings from the provided excerpts.")
    jwstdoi: float = Field(..., description="Whether the JWST data is accompanied by a DOI (specifically 10.17909 prefix)")
    reason: str = Field(..., description="Justification for the given 'jwstdoi' score based ONLY on the provided excerpts")


class JWSTPreprintDOIAnalyzer:
    SCIENCE_KEYWORDS = [
        "jwst", "james webb space telescope", "webb telescope", "webb",
        "nircam", "nirspec", "miri", "niriss", "fgs",
        # Add common ways observations might be described
        "jwst observation", "webb observation", "program id", "proposal id",
    ]
    DOI_KEYWORDS = [
        "mast", "mast archive",
        "10.17909", # Specific prefix for JWST DOIs
        # "doi", "digital object identifier", # this might trigger in all the references
        "data availability", "acknowledgments", 
        "program id", "proposal id",
    ]
    # combine and lowercase for efficient searching
    ALL_KEYWORDS_LOWER = sorted(list(set(
        [k.lower() for k in SCIENCE_KEYWORDS] + [k.lower() for k in DOI_KEYWORDS]
    )), key=len, reverse=True) # Sort by length to match longer phrases first
    SCIENCE_KEYWORDS_LOWER = sorted(list(set(
        [k.lower() for k in SCIENCE_KEYWORDS]
    )), key=len, reverse=True)
    DOI_KEYWORDS_LOWER = sorted(list(set(
        [k.lower() for k in DOI_KEYWORDS]
    )), key=len, reverse=True)


    def __init__(self,
                 year_month: str,  # Format: YYYY-MM
                 output_dir: Path,
                 science_threshold: float = 0.5,
                 doi_threshold: float = 0.8,
                 ads_key: Optional[str] = None,
                 openai_key: Optional[str] = None,
                 cohere_key: Optional[str] = None, # <-- New: Cohere Key
                 reranker_model: str = 'rerank-v3.5', # <-- New: Reranker model
                 top_k_snippets: int = 15, # <-- New: Number of snippets for LLM
                 context_sentences: int = 3, # <-- New: Sentences before/after keyword sentence
                 validate_llm: bool = False, # <-- New: Flag for LLM validation step
                 reprocess: bool = False,
                 ):
        """Initialize the JWST paper analyzer."""
        self.year_month = year_month
        self.reprocess = reprocess
        self.science_threshold = science_threshold
        self.doi_threshold = doi_threshold
        self.reranker_model = reranker_model
        self.top_k_snippets = top_k_snippets
        self.context_sentences = context_sentences
        self.validate_llm = validate_llm # Store validation flag


        # Setup API keys
        self.ads_key = ads_key or os.getenv('ADS_API_KEY')
        self.openai_key = openai_key or os.getenv('OPENAI_API_KEY')
        self.cohere_key = cohere_key or os.getenv('COHERE_API_KEY') # <-- New
        if not self.ads_key or not self.openai_key:
            raise ValueError("Both ADS_API_KEY and OPENAI_API_KEY must be provided")

        self.openai_client = OpenAI(api_key=self.openai_key, max_retries=2) 

        if not self.cohere_key:
            logging.warning("COHERE_API_KEY not found. Reranking will be skipped (using original order).")
            self.cohere_client = None
        else:
            try:
                # Test connection during initialization
                self.cohere_client = cohere.ClientV2(self.cohere_key)
                # Perform a lightweight check if possible, e.g., model availability
                available_cohere_models = self.cohere_client.models.list()
                available_models = [m.name for m in available_cohere_models.models]
                if self.reranker_model not in available_models and not self.reranker_model.startswith('rerank-'):
                     logging.warning(f"Cohere reranker model '{self.reranker_model}' not found in available models. Check model name.")
                     # Optionally fall back or raise error
                     # self.cohere_client = None
            except Exception as e:
                logging.error(f"An unexpected error occurred during Cohere client initialization: {e}")
                self.cohere_client = None


        # Setup directories
        self.papers_dir = output_dir / "papers"
        self.texts_dir = output_dir / "texts"
        self.results_dir = output_dir / "results"
        self._setup_directories()

        # Setup cache files
        self.download_cache = self.results_dir / f"{self.year_month}_downloaded.json"
        self.science_cache = self.results_dir / f"{self.year_month}_science.json"
        self.doi_cache = self.results_dir / f"{self.year_month}_dois.json"
        self.skipped_cache = self.results_dir / f"{self.year_month}_skipped.json"
        self.snippets_cache = self.results_dir / f"{self.year_month}_snippets.json"

        # Ensure NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except (nltk.downloader.DownloadError, LookupError):
            logging.info("NLTK 'punkt' tokenizer not found. Attempting download...")
            try:
                nltk.download('punkt', quiet=True)
                nltk.data.find('tokenizers/punkt') # Verify download
                logging.info("'punkt' tokenizer downloaded successfully.")
            except Exception as e:
                logging.error(f"Failed to download NLTK 'punkt' tokenizer: {e}")
                logging.error("Please install it manually: run `python -c \"import nltk; nltk.download('punkt')\"`")
                sys.exit(1)


    # --- Directory and Cache Methods (Unchanged) ---
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.papers_dir, self.texts_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_cache(self, cache_file: Path) -> Dict:
        """Load a cache file if it exists."""
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Cache file {cache_file} is corrupted. Starting fresh.")
                # Optionally backup corrupted file
                # corrupted_path = cache_file.with_suffix(f".corrupted_{int(time.time())}")
                # cache_file.rename(corrupted_path)
                # logging.warning(f"Corrupted cache moved to {corrupted_path}")
                return {}
            except Exception as e:
                logging.error(f"Failed to load cache file {cache_file}: {e}")
                return {} # Treat as empty on other errors too
        return {}

    def _save_cache(self, cache_file: Path, data: Dict):
        """Save data to a cache file."""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save cache file {cache_file}: {e}")


    # --- Paper Acquisition Methods (Minor change in ADS query logic) ---
    def _get_paper_list(self) -> List[Dict[str, str]]:
        """Fetch list of astronomy papers for the specified month from ADS."""
        logging.info(f"Fetching paper list for {self.year_month}")

        # Validate year_month format before using it
        try:
            year_full, month = self.year_month.split('-')
            if len(year_full) != 4 or not year_full.isdigit() or len(month) != 2 or not month.isdigit():
                raise ValueError("Invalid format")
            year_short = year_full[2:] # Get the last two digits of the year (YY)
            arxiv_pattern = f"arXiv:{year_short}{month}.*"
        except ValueError:
            logging.error("`year_month` argument format must be YYYY-MM")
            raise ValueError("`year_month` argument format must be YYYY-MM")

        url = "https://api.adsabs.harvard.edu/v1/search/query"
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

        try:
            response = requests.get(url, params=params, headers=headers, timeout=30) # Add timeout
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logging.error("ADS API request timed out.")
            return []
        except requests.exceptions.RequestException as e:
            logging.error(f"ADS API request failed: {e}")
            # Check for specific status codes if needed (e.g., 401 Unauthorized)
            if hasattr(e, 'response') and e.response is not None:
                 logging.error(f"ADS Response Status: {e.response.status_code}")
                 logging.error(f"ADS Response Body: {e.response.text[:500]}") # Log part of the body
                 if e.response.status_code == 401:
                     logging.error("Check if your ADS_API_KEY is valid.")
            return [] # Return empty list on failure

        try:
            data = response.json()
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON response from ADS: {response.text[:500]}")
            return []

        papers = []
        if 'response' not in data or 'docs' not in data['response']:
             logging.warning(f"Unexpected ADS response format for {self.year_month}")
             return []

        for doc in data['response']['docs']:
            # Ensure 'identifier' exists and is a list
            identifiers = doc.get('identifier', [])
            if not isinstance(identifiers, list):
                identifiers = [identifiers] # Handle case where it might not be a list

            arxiv_id = next(
                (id_str.split(':')[1] for id_str in identifiers
                 if isinstance(id_str, str) and id_str.startswith('arXiv:')),
                None
            )
            # Basic validation of arxiv ID format
            if arxiv_id and re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", arxiv_id.split('/')[-1]):
                if 'bibcode' in doc:
                    papers.append({
                        'arxiv_id': arxiv_id,
                        'bibcode': doc['bibcode']
                    })
            elif arxiv_id:
                 logging.warning(f"Extracted potential arXiv ID '{arxiv_id}' has unexpected format. Skipping.")

        logging.info(f"Found {len(papers)} valid papers")
        return papers

    # --- Download/Convert/Skip Methods ---
    def _is_downloaded_and_converted(self, paper: Dict[str, str]) -> bool:
        """Check if paper PDF is downloaded AND text file exists."""
        downloaded_cache = self._load_cache(self.download_cache)
        txt_path = self.texts_dir / f"{paper['arxiv_id']}.txt"
        pdf_path = self.papers_dir / f"{paper['arxiv_id']}.pdf"
        # Check cache, pdf existence, and text existence
        return (not self.reprocess
                and paper['arxiv_id'] in downloaded_cache
                and pdf_path.exists()
                and txt_path.exists())

    def _is_skipped(self, paper: Dict[str, str]) -> bool:
        """Check if paper was previously skipped."""
        skipped = self._load_cache(self.skipped_cache)
        return not self.reprocess and paper['arxiv_id'] in skipped

    def _mark_as_skipped(self, paper: Dict[str, str], reason: str):
        """Mark a paper as skipped with the given reason."""
        arxiv_id = paper['arxiv_id']
        logging.warning(f"Skipping paper {arxiv_id}: {reason}")
        skipped = self._load_cache(self.skipped_cache)
        skipped[arxiv_id] = {
            "reason": reason,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_cache(self.skipped_cache, skipped)
        # Also ensure it's not marked as downloaded if we skip it now
        downloaded = self._load_cache(self.download_cache)
        if arxiv_id in downloaded:
            del downloaded[arxiv_id]
            self._save_cache(self.download_cache, downloaded)


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
        """Convert PDF to text using pdftext. Returns True if successful, False otherwise."""
        arxiv_id = paper['arxiv_id']
        pdf_path = self.papers_dir / f"{arxiv_id}.pdf"
        txt_path = self.texts_dir / f"{arxiv_id}.txt"

        if txt_path.exists() and not self.reprocess:
            # logging.info(f"Text file already exists for {arxiv_id}")
            return True

        if not pdf_path.exists():
            logging.warning(f"Cannot convert {arxiv_id}: PDF file not found at {pdf_path}")
            self._mark_as_skipped(paper, "PDF not found for conversion")
            return False

        logging.info(f"Converting paper {arxiv_id} to text")
        try:
            # Use timeout to prevent hangs on problematic PDFs
            result = subprocess.run(
                ["pdftext", "--sort", str(pdf_path), "--out_path", str(txt_path)],
                check=True, # Raise CalledProcessError on non-zero exit
                capture_output=True,
                text=True,
                encoding='utf-8', # Specify encoding for captured output
                timeout=2 
            )
            # Check if the output file was actually created and is not empty
            if not txt_path.exists() or txt_path.stat().st_size == 0:
                 logging.error(f"pdftext ran for {arxiv_id}, but output file is missing or empty.")
                 logging.error(f"pdftext stdout: {result.stdout}")
                 logging.error(f"pdftext stderr: {result.stderr}")
                 self._mark_as_skipped(paper, "PDF conversion produced empty/missing file")
                 # Clean up empty file
                 if txt_path.exists(): txt_path.unlink()
                 return False

            # logging.debug(f"pdftext stdout for {arxiv_id}: {result.stdout}") # Optional: log stdout on success
            return True

        except subprocess.TimeoutExpired:
             logging.error(f"Timeout converting {arxiv_id} with pdftext.")
             self._mark_as_skipped(paper, "PDF conversion timed out")
             # Clean up potentially partial text file
             if txt_path.exists(): txt_path.unlink()
             return False
        except subprocess.CalledProcessError as e:
            # Log more details from the error
            logging.error(f"Error converting {arxiv_id} (pdftext exit code {e.returncode}):")
            logging.error(f"  Command: {e.cmd}")
            # Decode stderr carefully in case of weird characters
            stderr_decoded = e.stderr.decode('utf-8', errors='replace') if isinstance(e.stderr, bytes) else e.stderr
            logging.error(f"  Stderr: {stderr_decoded}")
            self._mark_as_skipped(paper, f"PDF conversion failed (pdftext error): {stderr_decoded[:200]}") # Limit reason length
            # Clean up potentially partial text file
            if txt_path.exists(): txt_path.unlink()
            return False
        except FileNotFoundError:
            logging.error("`pdftext` command not found. Is it installed and in the system PATH?")
            # Skip this paper and potentially exit or mark all subsequent as skipped
            self._mark_as_skipped(paper, "pdftext command not found")
            # Consider raising an exception here to stop the whole process if pdftext is essential
            # raise RuntimeError("pdftext command not found. Aborting.")
            return False
        except Exception as e:
            logging.error(f"Unexpected error converting {arxiv_id}: {str(e)}")
            self._mark_as_skipped(paper, f"Unexpected PDF conversion error: {str(e)}")
            if txt_path.exists(): txt_path.unlink()
            return False

    # --- New Snippet Extraction and Reranking Methods ---

    def _extract_relevant_snippets(self, paper_text: str, keywords: List[str]) -> List[str]:
        """Extract sentences containing keywords, plus surrounding context sentences."""
        if not paper_text:
            return []

        # preprocess text:
        # \n\n -> \n; \n -> nothing; remove multiple spaces
        paper_text = re.sub(r'\n{2,}', '\n\n', paper_text)
        paper_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', paper_text)
        paper_text = re.sub(r' {2,}', ' ', paper_text)

        try:
            sentences = sent_tokenize(paper_text)
        except Exception as e:
            logging.warning(f"NLTK sentence tokenization failed: {e}. Falling back to simple split.")
            sentences = [line for line in paper_text.splitlines() if line.strip()]

        if not sentences:
            return []

        relevant_indices: Set[int] = set()
        extracted_snippets: Set[str] = set()

        # Find indices of sentences containing keywords
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            # Use the pre-sorted keywords list (longest first)
            for keyword in keywords:
                if keyword in sentence_lower:
                    relevant_indices.add(i)
                    break # Move to next sentence once a keyword is found

        # Expand context and collect sentences
        final_indices: Set[int] = set()
        for i in relevant_indices:
            start = max(0, i - self.context_sentences)
            end = min(len(sentences), i + self.context_sentences + 1)
            for j in range(start, end):
                final_indices.add(j)

        # Create snippets from contiguous blocks of selected sentences
        if not final_indices:
            return []

        sorted_indices = sorted(list(final_indices))
        current_snippet_sentences = []
        last_index = -2 # Initialize to ensure the first index starts a new block

        for index in sorted_indices:
            # If index is not contiguous with the last one, save the previous snippet and start new
            if index != last_index + 1 and current_snippet_sentences:
                extracted_snippets.add(" ".join(current_snippet_sentences).strip())
                current_snippet_sentences = []

            current_snippet_sentences.append(sentences[index])
            last_index = index

        # Add the last collected snippet
        if current_snippet_sentences:
            extracted_snippets.add(" ".join(current_snippet_sentences).strip())

        # logging.debug(f"Extracted {len(extracted_snippets)} unique snippets.")
        return list(extracted_snippets)


    def _rerank_snippets(self, query: str, snippets: List[str]) -> List[Dict[str, Union[str, Optional[float]]]]:
        """Rerank snippets using Cohere API based on the query. Returns list of {'snippet': str, 'score': float}."""
        if not self.cohere_client or not snippets:
            # If no client or no snippets, return original order (or top N) with None scores
            # return snippets[:self.top_k_snippets] # <--- Delete this line
            # Add this line:
            return [{'snippet': s, 'score': None} for s in snippets[:self.top_k_snippets]]

        logging.debug(f"Reranking {len(snippets)} snippets with Cohere model '{self.reranker_model}' for query: '{query}'")
        try:
            # Ensure snippets are not empty strings... (rest of the try block preamble is the same)
            non_empty_snippets = [s for s in snippets if s and s.strip()]
            if not non_empty_snippets:
                 logging.warning("All extracted snippets were empty after stripping.")
                 # return [] # <--- Delete this line
                 return [] # Return empty list consistent with new type

            # ... (limit max_docs_for_rerank check remains the same) ...

            results = self.cohere_client.rerank(
                 query=query,
                 documents=non_empty_snippets,
                 top_n=self.top_k_snippets,
                 model=self.reranker_model
            )

            # Extract the reranked documents AND scores (around line 500)
            # reranked_docs = [snippets[result.index] for result in results.results] # <--- Delete this line
            # Add this line:
            reranked_data = [{'snippet': non_empty_snippets[result.index], 'score': result.relevance_score} for result in results.results]

            logging.info(f"Reranked scores (top 3): {[f'{r.relevance_score:.3f}' for r in results.results[:3]]}")

            # ... (time.sleep remains commented out) ...
            # return reranked_docs # <--- Delete this line
            # Add this line:
            return reranked_data

        except Exception as e:
            logging.error(f"Unexpected error during Cohere reranking: {e}")
            # Fallback: Return top K original snippets with None scores
            # return non_empty_snippets[:self.top_k_snippets] # <--- Delete this line
            # Add this line:
            return [{'snippet': s, 'score': None} for s in non_empty_snippets[:self.top_k_snippets]]

    # --- Analysis Methods (Modified) ---

    def _needs_analysis(self, paper: Dict[str, str], cache_file: Path) -> bool:
        """Generic check if paper needs analysis based on cache and reprocess flag."""
        analyzed = self._load_cache(cache_file)
        return self.reprocess or paper['arxiv_id'] not in analyzed

    def _call_openai_parse(self, model_name: str, system_prompt: str, user_prompt: str, response_model: BaseModel) -> Optional[Dict]:
        """Helper function to call OpenAI parse API with error handling."""
        try:
            result = self.openai_client.beta.chat.completions.parse(
                model=model_name, # e.g., "gpt-4o-mini-2024-07-18"
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=response_model,
                timeout=90 # Set a timeout for the API call
            )
            # Accessing the parsed Pydantic model instance
            # Note: The structure might change slightly depending on the library version.
            # Assuming it's accessible via `.choices[0].message.tool_calls[0].function.parsed_arguments` or similar path
            # Let's assume the `parse` method itself returns the Pydantic object directly or within a standard structure.
            # Based on common usage, let's try accessing via the message content which should now be the structured JSON
            # if response_format=PydanticModel is used correctly.

            # The `.parse()` method is designed to return the Pydantic model instance directly.
            parsed_object = result.choices[0].message.parsed # Access the parsed Pydantic object
            if parsed_object:
                 # Convert Pydantic model back to dict for consistent handling/caching
                 return parsed_object.model_dump()
            else:
                 # This case might indicate an issue with parsing or an unexpected response structure
                 logging.error("OpenAI response parsed, but no Pydantic object found in expected location.")
                 logging.debug(f"Full OpenAI response object: {result}")
                 # Attempt to manually parse if content is JSON string
                 try:
                      raw_content = result.choices[0].message.content
                      if isinstance(raw_content, str):
                           parsed_json = json.loads(raw_content)
                           # Validate with Pydantic model again
                           validated_model = response_model.model_validate(parsed_json)
                           return validated_model.model_dump()
                      else:
                           logging.error("Raw message content is not a string for manual parsing.")
                           return None
                 except (json.JSONDecodeError, ValidationError, Exception) as manual_parse_err:
                      logging.error(f"Manual parsing/validation of OpenAI response failed: {manual_parse_err}")
                      return None


        except BadRequestError as e:
             # This error now less likely for token limits, but could occur for other reasons (e.g., bad input format)
            logging.warning(f"OpenAI API BadRequestError: {e}")
            # Distinguish token limit if possible, although less likely now
            if "context length" in str(e).lower():
                 logging.warning(f"Snippets might still exceed token limit: {e}")
                 # Return a specific error structure or None
                 return {"error": "token_limit", "message": str(e)}
            else:
                 return {"error": "bad_request", "message": str(e)}

        except Exception as e:
            logging.error(f"OpenAI API call failed: {e}")
            return {"error": "api_error", "message": str(e)}

        # Should not be reached if parse worked correctly
        return None

    def _analyze_science(self, paper: Dict[str, str]) -> Dict:
        """Analyze paper for JWST science content using extracted snippets."""
        arxiv_id = paper['arxiv_id']
        logging.info(f"Analyzing JWST science content for {arxiv_id}")
        science_results = self._load_cache(self.science_cache)

        txt_path = self.texts_dir / f"{arxiv_id}.txt"
        if not txt_path.exists():
             logging.warning(f"Text file not found for {arxiv_id}, cannot analyze science.")
             return {"jwstscience": -1.0, "reason": "Analysis failed: Text file missing", "quotes": []} # Indicate failure

        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                paper_text = f.read()
        except Exception as e:
            logging.error(f"Failed to read text file {txt_path}: {e}")
            return {"jwstscience": -1.0, "reason": "Analysis failed: Cannot read text file", "quotes": []}

        # 1. Extract Snippets
        # Use science keywords, potentially adding DOI keywords if context helps
        keywords_to_find = self.SCIENCE_KEYWORDS_LOWER # Focus on science terms primarily
        all_snippets = self._extract_relevant_snippets(paper_text, keywords_to_find)

        if not all_snippets:
            logging.info(f"No relevant keywords found for science analysis in {arxiv_id}.")
            result = {"jwstscience": 0.0, "quotes": [], "reason": "No relevant keywords (e.g., JWST, instruments) found in text."}
            science_results[arxiv_id] = result
            self._save_cache(self.science_cache, science_results)
            return result

        # 2. Rerank Snippets
        rerank_query = "Do these text snippets originate from a paper that presents new scientific results derived from James Webb Space Telescope (JWST) data? Mentions of previous or future JWST observations do not count."
        reranked_data = self._rerank_snippets(rerank_query, all_snippets)

        if not reranked_data:
             logging.warning(f"Reranking produced no snippets for {arxiv_id}. Skipping LLM analysis.")
             result = {"jwstscience": 0.0, "quotes": [], "reason": "Keyword snippets found but none survived reranking/filtering."}
             science_results[arxiv_id] = result
             self._save_cache(self.science_cache, science_results)
             return result

        try:
            snippets_cache_data = self._load_cache(self.snippets_cache)
            # Ensure the entry for this arxiv_id exists
            if arxiv_id not in snippets_cache_data:
                 snippets_cache_data[arxiv_id] = {}
            # Store the reranked snippets and scores
            snippets_cache_data[arxiv_id]["science_analysis"] = reranked_data
            self._save_cache(self.snippets_cache, snippets_cache_data)
        except Exception as e:
            logging.error(f"Failed to save science snippets cache for {arxiv_id}: {e}")

        # 3. Prepare LLM Input
        reranked_snippets_for_llm = [item['snippet'] for item in reranked_data]
        snippets_text = "\n---\n".join([f"Excerpt {i+1}:\n{s}" for i, s in enumerate(reranked_snippets_for_llm)])

        # Limit total snippet length sent to LLM just in case (e.g., ~30k chars for ~8k tokens)
        # logging.info(f"Full snippet:\n{snippets_text}")
        max_chars = 30000
        if len(snippets_text) > max_chars:
             logging.warning(f"Total snippet text for {arxiv_id} exceeds {max_chars} chars, truncating.")
             snippets_text = snippets_text[:max_chars]


        system_prompt = "You are an expert astronomy researcher analyzing paper excerpts to determine if they present new JWST science. Focus ONLY on the provided excerpts. Base your score and reason entirely on whether these excerpts indicate the use of actual JWST observational data (not simulations, proposals, or future work) to derive new scientific findings. Comparisons to previous JWST work count if used to establish a new result in *this* paper. Score 0 if no such evidence exists in the excerpts."
        user_prompt = f"""
Analyze the following excerpts from an astronomy paper to determine the likelihood it presents new James Webb Space Telescope (JWST) science results.

**Scoring Guidelines:**
*   **0.0**: No mention of JWST/Webb or only mentions future plans, proposals, simulations, or funding without presenting data/results from it within these excerpts.
*   **0.1-0.3**: Mentions JWST/Webb or its instruments, maybe cites prior JWST work, but the excerpts don't clearly show *new* analysis or results from JWST in *this* paper. Could be purely motivational or comparative discussion.
*   **0.4-0.6**: Moderate indication that JWST data/results are used (e.g., mentions analyzing JWST data, shows plots potentially from JWST), but the excerpts lack definitive statements of *new* findings derived from it. Cannot simply cite other JWST papers.
*   **0.7-0.9**: Strong indication from the excerpts that *new* JWST observations are presented or analyzed to derive new scientific results in this paper, even if new JWST data are presented just for a single table or plot.
*   **1.0**: Excerpts explicitly state that new JWST observations/data (e.g., from NIRCam, NIRSpec, MIRI, NIRISS) were obtained/analyzed for this work and new results are presented.

**Task:**
Determine whether *this paper* is a JWST science paper. Return a JSON object adhering to the specified format. The 'quotes' MUST be exact substrings copied from the provided excerpts below.

**Examples:**
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

**Excerpts:**
{snippets_text}

**JSON Output Format:**
{{
    "quotes": ["Exact quote 1 from excerpts...", "Exact quote 2..."],
    "jwstscience": <float between 0.0 and 1.0>,
    "reason": "Justification based ONLY on the excerpts."
}}
"""

        # 4. Call LLM (Initial Analysis)
        # Use a capable but potentially cheaper model if appropriate for snippets
        model_to_use = "gpt-4o-mini-2024-07-18" 
        llm_result = self._call_openai_parse(model_to_use, system_prompt, user_prompt, JWSTScienceLabelerModel)

        # Handle LLM call errors
        if llm_result is None or "error" in llm_result:
             error_reason = f"LLM analysis failed: {llm_result['message'] if llm_result else 'Unknown error'}"
             self._mark_as_skipped(paper, error_reason)
             # Return failure indicator, do not cache partial result
             return {"jwstscience": -1.0, "reason": error_reason, "quotes": []}

        # Use the result from the first call if validation is disabled or score is low
        final_result = llm_result

        # 5. Optional Validation Step (If enabled and score > 0)
        # Note: Validation may be less critical now with focused snippets. Evaluate its usefulness.
        if self.validate_llm and final_result.get("jwstscience", 0.0) > 0:
            logging.debug(f"Performing LLM validation for science score on {arxiv_id}")
            validation_system_prompt = "You are validating a previous analysis of whether paper excerpts indicate new JWST science. Review the original quotes, reason, and score. Keep the quotes EXACTLY the same. Adjust the score (0.0-1.0) and reason ONLY if the original score seems inconsistent with the provided quotes based on the scoring guidelines (new JWST science results presented in *this* paper based *only* on these quotes)."
            validation_user_prompt = f"""
**Original Analysis:**
Quotes: {json.dumps(final_result.get("quotes", []), indent=2)}
Score: {final_result.get("jwstscience", "N/A")}
Reason: "{final_result.get("reason", "N/A")}"

**Scoring Guidelines (Reminder):**
*   0.0: No JWST science evidence in quotes.
*   0.1-0.3: Mentions JWST, but no clear new science from quotes.
*   0.4-0.6: Moderate indication of JWST data use, lacks definitive new findings in quotes.
*   0.7-0.9: Strong indication of new JWST science derived in quotes.
*   1.0: Explicit statement of new JWST observations/analysis/results in quotes.

**Task:**
Validate the score based *only* on the provided quotes and guidelines. Return JSON with potentially revised score/reason, but IDENTICAL quotes.

**Task:**
 Return a JSON object adhering to the specified format. The 'quotes' MUST be exact substrings copied from the provided excerpts below.

**Examples:**
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

**JSON Output Format:**
{{
    "quotes": {json.dumps(final_result.get("quotes", []))}, // MUST BE IDENTICAL
    "jwstscience": <float between 0.0 and 1.0>,
    "reason": "Validated justification based ONLY on the original quotes."
}}
"""
            # Use a potentially stronger/different model for validation? Or same model? Let's use the same for now.
            validation_llm_result = self._call_openai_parse(model_to_use, validation_system_prompt, validation_user_prompt, JWSTScienceLabelerModel)

            if validation_llm_result and "error" not in validation_llm_result:
                 # Ensure quotes haven't changed (LLMs might still hallucinate)
                 if validation_llm_result.get("quotes") == final_result.get("quotes"):
                      final_result = validation_llm_result # Update with validated result
                 else:
                      logging.warning(f"LLM validation for {arxiv_id} changed quotes. Discarding validation.")
            elif validation_llm_result: # Handle error during validation
                 logging.warning(f"LLM validation step failed for {arxiv_id}: {validation_llm_result.get('message', 'Unknown error')}")
                 # Keep the original result if validation fails

        # 6. Cache the final result
        science_results = self._load_cache(self.science_cache) # Reload cache in case of concurrent runs (though this script isn't concurrent yet)
        science_results[arxiv_id] = final_result
        self._save_cache(self.science_cache, science_results)

        return final_result


    def _analyze_doi(self, paper: Dict[str, str]) -> Dict:
        """Analyze paper for JWST DOIs using extracted snippets."""
        arxiv_id = paper['arxiv_id']
        logging.info(f"Analyzing JWST DOIs for {arxiv_id}")
        doi_results = self._load_cache(self.doi_cache)

        txt_path = self.texts_dir / f"{arxiv_id}.txt"
        if not txt_path.exists():
             logging.warning(f"Text file not found for {arxiv_id}, cannot analyze DOI.")
             return {"jwstdoi": -1.0, "reason": "Analysis failed: Text file missing", "quotes": []}

        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                paper_text = f.read()
        except Exception as e:
            logging.error(f"Failed to read text file {txt_path}: {e}")
            return {"jwstdoi": -1.0, "reason": "Analysis failed: Cannot read text file", "quotes": []}

        # 1. Extract Snippets (Focus on DOI keywords and context)
        # Include science keywords as context for DOI might be near instrument mentions
        keywords_to_find = self.DOI_KEYWORDS_LOWER + [k for k in self.SCIENCE_KEYWORDS_LOWER if k not in self.DOI_KEYWORDS_LOWER] # Combine, avoid duplicates
        all_snippets = self._extract_relevant_snippets(paper_text, keywords_to_find)

        if not all_snippets:
            logging.info(f"No relevant keywords found for DOI analysis in {arxiv_id}.")
            result = {"jwstdoi": 0.0, "quotes": [], "reason": "No relevant keywords (e.g., DOI, 10.17909, data availability, JWST) found in text."}
            doi_results[arxiv_id] = result
            self._save_cache(self.doi_cache, doi_results)
            return result

        # 2. Rerank Snippets
        rerank_query = "Does this text mention a Digital Object Identifier (DOI) specifically for James Webb Space Telescope (JWST) data, which must start with 10.17909?"
        reranked_data = self._rerank_snippets(rerank_query, all_snippets) # <--- To this

        if not reranked_data:
             logging.warning(f"Reranking produced no snippets for DOI analysis {arxiv_id}. Skipping LLM.")
             result = {"jwstdoi": 0.0, "quotes": [], "reason": "Keyword snippets found but none survived reranking/filtering for DOI check."}
             doi_results[arxiv_id] = result
             self._save_cache(self.doi_cache, doi_results)
             return result

        try:
            snippets_cache_data = self._load_cache(self.snippets_cache)
            # Ensure the entry for this arxiv_id exists
            if arxiv_id not in snippets_cache_data:
                 snippets_cache_data[arxiv_id] = {}
            # Store the reranked snippets and scores
            snippets_cache_data[arxiv_id]["doi_analysis"] = reranked_data
            self._save_cache(self.snippets_cache, snippets_cache_data)
        except Exception as e:
            logging.error(f"Failed to save DOI snippets cache for {arxiv_id}: {e}")

        # 3. Prepare LLM Input
        reranked_snippets_for_llm = [item['snippet'] for item in reranked_data] 
        snippets_text = "\n---\n".join([f"Excerpt {i+1}:\n{s}" for i, s in enumerate(reranked_snippets_for_llm)]) 
        max_chars = 30000 # Same limit as science analysis
        if len(snippets_text) > max_chars:
             logging.warning(f"Total snippet text for DOI analysis {arxiv_id} exceeds {max_chars} chars, truncating.")
             snippets_text = snippets_text[:max_chars]

        system_prompt = "You are an expert astronomy researcher scanning paper excerpts for specific Digital Object Identifiers (DOIs) related to JWST data. Focus ONLY on the provided excerpts. Look for DOIs starting with '10.17909/' and check if the surrounding text links them explicitly to the JWST observations used in the paper. Other DOIs or mentions of program IDs without a DOI don't count for a high score."
        user_prompt = f"""
Analyze the following excerpts from an astronomy paper to determine if a JWST data DOI (specifically starting with '10.17909/') is cited for the data used in the paper.

**Scoring Guidelines:**
*   **0.0**: No mention of DOIs, '10.17909', or data availability sections in the excerpts.
*   **0.1**: Mentions JWST program IDs, acknowledgments, or MAST, but no DOI string is present in the excerpts.
*   **0.2-0.4**: Mentions DOIs, but none start with '10.17909/' OR a '10.17909/' DOI is mentioned but the excerpts don't clearly link it to the *JWST data* used in *this specific study* (could be for other data, software, or general archive).
*   **0.5-0.7**: A DOI starting with '10.17909/' is present, and the surrounding text weakly suggests it might be for the JWST data (e.g., appears in data section near JWST mention).
*   **0.8-1.0**: A DOI starting with '10.17909/' is present, and the excerpts explicitly state or strongly imply this DOI refers to the specific JWST observations/datasets analyzed in this paper (e.g., "The JWST data used... can be found at doi:10.17909/...")

**Task:**
Return a JSON object adhering to the specified format. The 'quotes' MUST be exact substrings copied from the provided excerpts below.

**Excerpts:**
{snippets_text}

**JSON Output Format:**
{{
    "quotes": ["Exact quote 1 from excerpts...", "Exact quote 2..."],
    "jwstdoi": <float between 0.0 and 1.0>,
    "reason": "Justification based ONLY on the excerpts and the DOI prefix '10.17909/'."
}}
"""

        # 4. Call LLM (Initial Analysis)
        model_to_use = "gpt-4o-mini-2024-07-18"
        llm_result = self._call_openai_parse(model_to_use, system_prompt, user_prompt, JWSTDOILabelerModel)

        # Handle LLM call errors
        if llm_result is None or "error" in llm_result:
             error_reason = f"LLM analysis failed for DOI: {llm_result['message'] if llm_result else 'Unknown error'}"
             # Don't mark as skipped here, as science analysis might have worked.
             # Just return failure for DOI part.
             logging.warning(f"DOI analysis failed for {arxiv_id}: {error_reason}")
             return {"jwstdoi": -1.0, "reason": error_reason, "quotes": []} # Indicate DOI failure

        final_result = llm_result

        # 5. Optional Validation Step
        if self.validate_llm and final_result.get("jwstdoi", 0.0) > 0:
            logging.debug(f"Performing LLM validation for DOI score on {arxiv_id}")
            validation_system_prompt = "You are validating a previous analysis of whether paper excerpts cite a JWST DOI (10.17909/). Review the original quotes, reason, and score. Keep the quotes EXACTLY the same. Adjust the score (0.0-1.0) and reason ONLY if the original score seems inconsistent with the provided quotes based on the scoring guidelines (explicit link between '10.17909/...' DOI and JWST data in the quotes)."
            validation_user_prompt = f"""
**Original Analysis:**
Quotes: {json.dumps(final_result.get("quotes", []), indent=2)}
Score: {final_result.get("jwstdoi", "N/A")}
Reason: "{final_result.get("reason", "N/A")}"

**Scoring Guidelines (Reminder):**
*   0.0: No DOI evidence.
*   0.1: Program ID/ack mentioned, no DOI string.
*   0.2-0.4: DOI present, but not '10.17909/' or not clearly linked to JWST data in quotes.
*   0.5-0.7: '10.17909/' DOI present, weak link to JWST data in quotes.
*   0.8-1.0: '10.17909/' DOI present, explicitly linked to JWST data in quotes.

**Task:**
Does this paper have a valid JWST DOI beginning with '10.17909/'? Validate the score based *only* on the provided quotes and guidelines. Return JSON with potentially revised score/reason, but IDENTICAL quotes.

**JSON Output Format:**
{{
    "quotes": {json.dumps(final_result.get("quotes", []))}, // MUST BE IDENTICAL
    "jwstdoi": <float between 0.0 and 1.0>,
    "reason": "Validated justification based ONLY on the original quotes and DOI prefix '10.17909/'."
}}
"""
            validation_llm_result = self._call_openai_parse(model_to_use, validation_system_prompt, validation_user_prompt, JWSTDOILabelerModel)

            if validation_llm_result and "error" not in validation_llm_result:
                 if validation_llm_result.get("quotes") == final_result.get("quotes"):
                      final_result = validation_llm_result
                 else:
                      logging.warning(f"LLM DOI validation for {arxiv_id} changed quotes. Discarding validation.")
            elif validation_llm_result:
                 logging.warning(f"LLM DOI validation step failed for {arxiv_id}: {validation_llm_result.get('message', 'Unknown error')}")

        # 6. Cache the final result
        doi_results = self._load_cache(self.doi_cache) # Reload cache
        doi_results[arxiv_id] = final_result
        self._save_cache(self.doi_cache, doi_results)

        return final_result

    # --- Reporting Method (Modified to handle potential -1 scores) ---
    def _generate_report(self):
        """Generate a summary report of the analysis."""
        science_results = self._load_cache(self.science_cache)
        doi_results = self._load_cache(self.doi_cache)
        skipped_results = self._load_cache(self.skipped_cache)
        downloaded_papers = self._load_cache(self.download_cache) # Get list of attempted downloads

        # Calculate total attempted (downloaded or skipped)
        total_attempted = len(downloaded_papers) + len(skipped_results)
        # Papers successfully processed (have a science score >= 0)
        successfully_processed = {k: v for k, v in science_results.items() if v.get("jwstscience", -1.0) >= 0}
        # Papers failing analysis (have science score < 0 or not in science_results but maybe in doi_results)
        analysis_failed_ids = set(science_results.keys()) - set(successfully_processed.keys())
        analysis_failed_count = len(analysis_failed_ids)

        # Count science papers based on threshold (only among successfully processed)
        science_papers_ids = {
            arxiv_id for arxiv_id, r in successfully_processed.items()
            if r.get("jwstscience", 0.0) >= self.science_threshold
        }
        science_papers_count = len(science_papers_ids)

        # Count DOIs only for papers identified as science papers
        papers_with_dois_count = 0
        papers_missing_dois_count = 0
        detailed_results_list = [] # Store detailed results for report

        for arxiv_id in science_papers_ids:
             doi_info = doi_results.get(arxiv_id)
             science_info = successfully_processed[arxiv_id] # Should exist
             doi_score = 0.0 # Default if DOI analysis failed or wasn't run
             doi_reason = "DOI analysis not available or failed"
             doi_quotes: List[str] = []

             if doi_info and doi_info.get("jwstdoi", -1.0) >= 0:
                  # DOI analysis ran successfully
                  doi_score = doi_info.get("jwstdoi", 0.0)
                  doi_reason = doi_info.get("reason", "N/A")
                  doi_quotes = doi_info.get("quotes", [])
                  if doi_score >= self.doi_threshold:
                       papers_with_dois_count += 1
                  else:
                       papers_missing_dois_count += 1
             else:
                 # DOI analysis failed or skipped for this science paper
                 papers_missing_dois_count += 1
                 if doi_info and doi_info.get("jwstdoi", -1.0) < 0:
                     doi_reason = doi_info.get("reason", "DOI analysis failed")


             detailed_results_list.append({
                 "arxiv_id": arxiv_id,
                 "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
                 "science_score": science_info.get("jwstscience"),
                 "science_reason": science_info.get("reason"),
                 "science_quotes": science_info.get("quotes"),
                 "doi_score": doi_score,
                 "doi_reason": doi_reason,
                 "doi_quotes": doi_quotes,
                 "has_valid_doi": doi_score >= self.doi_threshold
             })


        report = {
            "metadata": {
                "report_generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "year_month_analyzed": self.year_month,
                "science_threshold": self.science_threshold,
                "doi_threshold": self.doi_threshold,
                "reranker_model": self.reranker_model if self.cohere_client else "N/A (Cohere unavailable)",
                "top_k_snippets": self.top_k_snippets,
                "context_sentences": self.context_sentences,
                "llm_validation_enabled": self.validate_llm,
            },
            "summary": {
                "total_papers_identified_from_ads": total_attempted, # How many we tried to get
                "papers_downloaded_and_converted": len(successfully_processed) + analysis_failed_count, # PDFs downloaded+converted
                "papers_skipped_before_analysis": len(skipped_results), # Failed download/conversion etc.
                "papers_analysis_failed": analysis_failed_count, # Failed LLM calls etc.
                "papers_successfully_analyzed": len(successfully_processed),
                "jwst_science_papers_found": science_papers_count,
                "science_papers_with_valid_doi": papers_with_dois_count,
                "science_papers_missing_valid_doi": papers_missing_dois_count,
            },
            "skipped_papers_details": skipped_results,
            "jwst_science_papers_details": sorted(detailed_results_list, key=lambda x: x['arxiv_id']) # Sort by ID
        }

        report_path = self.results_dir / f"{self.year_month}_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logging.info(f"Report generated: {report_path}")
        logging.info(f"--- Summary for {self.year_month} ---")
        logging.info(f"  Total papers from ADS: {report['summary']['total_papers_identified_from_ads']}")
        logging.info(f"  Skipped (Download/Convert): {report['summary']['papers_skipped_before_analysis']}")
        logging.info(f"  Analysis Failed (LLM/etc): {report['summary']['papers_analysis_failed']}")
        logging.info(f"  Successfully Analyzed: {report['summary']['papers_successfully_analyzed']}")
        logging.info(f"  -> JWST Science Papers (Score >= {self.science_threshold}): {science_papers_count}")
        logging.info(f"    -> With Valid DOI (Score >= {self.doi_threshold}): {papers_with_dois_count}")
        logging.info(f"    -> Missing Valid DOI: {papers_missing_dois_count}")
        logging.info("--- End Summary ---")

        return report


    # --- Main Execution Pipeline ---
    def run(self):
        """Main execution pipeline"""
        start_time = time.time()
        logging.info(f"Starting analysis for {self.year_month}...")

        try:
            # 1. Get paper list from ADS
            papers = self._get_paper_list()
            if not papers:
                 logging.warning("No papers found or ADS query failed. Exiting.")
                 return # Exit if no papers to process

            # 2. Process each paper: Download -> Convert -> Analyze
            processed_ids = set() # Keep track of papers that reach analysis stage

            for i, paper in enumerate(papers):
                arxiv_id = paper['arxiv_id']
                logging.info(f"Processing paper {i+1}/{len(papers)}: {arxiv_id}")

                if self._is_skipped(paper):
                    logging.info(f"Paper {arxiv_id} was previously skipped. Skipping.")
                    continue

                # 2a. Download PDF
                if not (self.papers_dir / f"{arxiv_id}.pdf").exists() or self.reprocess:
                     if not self._download_paper(paper):
                          continue # Skip if download fails

                # 2b. Convert PDF to Text
                if not (self.texts_dir / f"{arxiv_id}.txt").exists() or self.reprocess:
                     if not self._convert_to_text(paper):
                          continue # Skip if conversion fails

                # --- If download and conversion successful, proceed to analysis ---
                processed_ids.add(arxiv_id)

                # 3. Analyze for JWST Science
                science_result = None
                if self._needs_analysis(paper, self.science_cache):
                    science_result = self._analyze_science(paper)
                else:
                    # Load from cache if not reprocessing
                    science_result = self._load_cache(self.science_cache).get(arxiv_id)
                    if science_result:
                         logging.info(f"Using cached science analysis for {arxiv_id}")
                    else:
                         # Should not happen if needs_analysis is False, but handle defensively
                         logging.warning(f"Cache logic error: Science cache missing for {arxiv_id} despite not needing analysis. Re-analyzing.")
                         science_result = self._analyze_science(paper)


                # Check if science analysis failed or paper doesn't meet threshold
                # Use .get() for safe access, default to -1.0 if key missing or result is None
                current_science_score = science_result.get("jwstscience", -1.0) if science_result else -1.0

                if current_science_score < self.science_threshold:
                    logging.info(f"Paper {arxiv_id} does not meet science threshold ({current_science_score:.2f} < {self.science_threshold}). Skipping DOI analysis.")
                    continue # Skip DOI analysis if not a science paper or if analysis failed

                # 4. Analyze for DOIs (only if it's a science paper)
                if self._needs_analysis(paper, self.doi_cache):
                    self._analyze_doi(paper)
                else:
                     # Log if using cached DOI result
                     if self._load_cache(self.doi_cache).get(arxiv_id):
                           logging.info(f"Using cached DOI analysis for {arxiv_id}")
                     else:
                           # This case is possible if DOI analysis failed previously or was skipped
                           logging.warning(f"DOI analysis needed for {arxiv_id} but cache missing/incomplete. Re-analyzing.")
                           self._analyze_doi(paper)

                # Optional short delay between papers to be polite to APIs, although OpenAI/Cohere clients handle retries
                # time.sleep(0.2)


            # 5. Generate final summary report
            report = self._generate_report()

            end_time = time.time()
            logging.info(f"Analysis complete for {self.year_month} in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            logging.exception(f"An unhandled error occurred during the run: {e}") # Log stack trace
            # Optionally try to generate a partial report
            try:
                logging.info("Attempting to generate partial report after error...")
                self._generate_report()
            except Exception as report_err:
                logging.error(f"Failed to generate partial report: {report_err}")
            raise # Reraise the original exception


def main():
    parser = argparse.ArgumentParser(
        description="Download arXiv papers via ADS, extract text, rerank snippets, and use LLMs to classify JWST science content and DOI presence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument(
        "year_month",
        help="Month to analyze in YYYY-MM format (e.g., 2024-01)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./"), # Default to a dedicated subdir
        help="Directory to store all outputs (papers, texts, results)"
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
        "--reprocess",
        action="store_true",
        help="Force reprocessing of downloaded/analyzed papers"
    )
    # --- New Arguments ---
    parser.add_argument(
        "--top-k-snippets",
        type=int,
        default=15,
        help="Number of top reranked snippets to send to the LLM"
    )
    parser.add_argument(
        "--context-sentences",
        type=int,
        default=3,
        help="Number of sentences before and after a keyword sentence to include in a snippet"
    )
    parser.add_argument(
        "--reranker-model",
        default="rerank-v3.5",
        help="Cohere reranker model name"
    )
    parser.add_argument(
        "--validate-llm",
        action="store_true",
        help="Perform a second LLM call to validate the first analysis (increases cost/time)"
    )
    # --- API Keys ---
    parser.add_argument(
        "--ads-key",
        help="ADS API key (uses ADS_API_KEY env var if not provided)"
    )
    parser.add_argument(
        "--openai-key",
        help="OpenAI API key (uses OPENAI_API_KEY env var if not provided)"
    )
    parser.add_argument(
        "--cohere-key",
        help="Cohere API key (uses COHERE_API_KEY env var if not provided; reranking skipped if missing)"
    )


    args = parser.parse_args()

    # Validate thresholds
    if not 0 <= args.science_threshold <= 1:
        parser.error("Science threshold must be between 0 and 1")
    if not 0 <= args.doi_threshold <= 1:
        parser.error("DOI threshold must be between 0 and 1")

    # Validate year_month format superficially
    if not re.match(r"^\d{4}-\d{2}$", args.year_month):
         parser.error("year_month format must be YYYY-MM")

    # Create output directory if it doesn't exist
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create output directory {args.output_dir}: {e}")
        sys.exit(1)


    try:
        analyzer = JWSTPreprintDOIAnalyzer(
            year_month=args.year_month,
            output_dir=args.output_dir,
            science_threshold=args.science_threshold,
            doi_threshold=args.doi_threshold,
            ads_key=args.ads_key, # Will default to env var if None
            openai_key=args.openai_key, # Will default to env var if None
            cohere_key=args.cohere_key, # Will default to env var if None
            reranker_model=args.reranker_model,
            top_k_snippets=args.top_k_snippets,
            context_sentences=args.context_sentences,
            validate_llm=args.validate_llm,
            reprocess=args.reprocess,
        )
        analyzer.run()
    except ValueError as e: # Catch specific init errors like bad keys/format
        logging.error(f"Initialization Error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during execution: {e}")
        # The run method already logs exceptions, but we catch here for exit code
        sys.exit(1)

    logging.info("Script finished successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()