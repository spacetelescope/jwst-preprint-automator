import argparse
from pathlib import Path
import logging
import json
import os
import sys
import re
import time
from typing import Optional, List, Dict, Tuple, Set, Union, Any
import requests
import subprocess
from openai import OpenAI, BadRequestError
from pydantic import BaseModel, Field, ValidationError 
import nltk
import cohere 
from nltk.tokenize import sent_tokenize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        "jwst observation", "webb observation", "program id", "proposal id",
    ]
    DOI_KEYWORDS = [
        "mast", "mast archive",
        "10.17909", # Specific prefix for MAST
        # "doi", "digital object identifier", # this might trigger in all the references of ApJ papers
        "data availability", "acknowledgments",
        "program id", "proposal id",
    ]
    ALL_KEYWORDS_LOWER = sorted(list(set(
        [k.lower() for k in SCIENCE_KEYWORDS] + [k.lower() for k in DOI_KEYWORDS]
    )), key=len, reverse=True) # Sort by length to match longer phrases first
    SCIENCE_KEYWORDS_LOWER = sorted(list(set(
        [k.lower() for k in SCIENCE_KEYWORDS]
    )), key=len, reverse=True)
    DOI_KEYWORDS_LOWER = sorted(list(set(
        [k.lower() for k in DOI_KEYWORDS]
    )), key=len, reverse=True)

    PROMPT_FILES = {
        'science_system': 'science_system.txt',
        'science_user': 'science_user.txt',
        'science_validate_system': 'science_validate_system.txt',
        'science_validate_user': 'science_validate_user.txt',
        'rerank_science_query': 'rerank_science_query.txt',
        'doi_system': 'doi_system.txt',
        'doi_user': 'doi_user.txt',
        'doi_validate_system': 'doi_validate_system.txt',
        'doi_validate_user': 'doi_validate_user.txt',
        'rerank_doi_query': 'rerank_doi_query.txt',
    }

    def __init__(self,
                 output_dir: Path,
                 year_month: Optional[str] = None, 
                 arxiv_id: Optional[str] = None, 
                 prompts_dir: Path = Path("./prompts"),
                 science_threshold: float = 0.5,
                 doi_threshold: float = 0.8,
                 reranker_threshold: float = 0.1, # skip LLM calls if reranker below this threshold
                 ads_key: Optional[str] = None,
                 openai_key: Optional[str] = None,
                 cohere_key: Optional[str] = None,
                 gpt_model: str = 'gpt-4.1-mini-2025-04-14', # 'gpt-4o-mini-2024-07-18',
                 reranker_model: str = 'rerank-v3.5', 
                 top_k_snippets: int = 15,
                 context_sentences: int = 3,
                 validate_llm: bool = False,
                 reprocess: bool = False,
                 ):
        """Initialize the JWST paper analyzer."""

        if not year_month and not arxiv_id:
             raise ValueError("Either 'year_month' or 'arxiv_id' must be provided.")
        if year_month and arxiv_id:
             raise ValueError("Provide either 'year_month' or 'arxiv_id', not both.")

        self.year_month = year_month # Will be None in single mode
        self.single_arxiv_id = arxiv_id # Will be None in batch mode
        self.run_mode = "batch" if year_month else "single"

        self.reprocess = reprocess
        self.science_threshold = science_threshold
        self.doi_threshold = doi_threshold
        self.reranker_threshold = reranker_threshold
        self.gpt_model = gpt_model
        self.reranker_model = reranker_model 
        self.top_k_snippets = top_k_snippets
        self.context_sentences = context_sentences
        self.validate_llm = validate_llm


        # Setup API keys
        self.ads_key = ads_key or os.getenv('ADS_API_KEY')
        self.openai_key = openai_key or os.getenv('OPENAI_API_KEY')
        self.cohere_key = cohere_key or os.getenv('COHERE_API_KEY')
        if self.run_mode == "batch" and not self.ads_key:
             logging.warning("ADS_API_KEY not provided. Batch mode ('year_month') will fail.")
        if not self.openai_key:
             raise ValueError("OPENAI_API_KEY must be provided (as argument or environment variable)")

        self.openai_client = OpenAI(api_key=self.openai_key, max_retries=2)

        if not self.cohere_key:
            logging.warning("COHERE_API_KEY not found. Reranking will be skipped (using original order).")
            self.cohere_client = None
        else:
            try:
                self.cohere_client = cohere.ClientV2(self.cohere_key)
                available_cohere_models = self.cohere_client.models.list()
                available_models = [m.name for m in available_cohere_models.models]
                if self.reranker_model not in available_models and not self.reranker_model.startswith('rerank-'):
                         logging.warning(f"Cohere reranker model '{self.reranker_model}' not found in available models. Check model name.")
            except Exception as e:
                 logging.error(f"An unexpected error occurred during Cohere client initialization: {e}")
                 self.cohere_client = None


        # create directories and cache files
        self.output_dir = output_dir 
        self.papers_dir = output_dir / "papers"
        self.texts_dir = output_dir / "texts"
        self.results_dir = output_dir / "results"
        self.prompts_dir = prompts_dir
        self._setup_directories() 

        # use year_month for batch mode, use a generic prefix or arxiv_id for single mode checks if needed
        cache_prefix = self.year_month if self.run_mode == "batch" else self.single_arxiv_id if self.single_arxiv_id else "single_run"
        self.download_cache = self.results_dir / f"{cache_prefix}_downloaded.json"
        self.science_cache = self.results_dir / f"{cache_prefix}_science.json"
        self.doi_cache = self.results_dir / f"{cache_prefix}_dois.json"
        self.skipped_cache = self.results_dir / f"{cache_prefix}_skipped.json"
        self.snippets_cache = self.results_dir / f"{cache_prefix}_snippets.json"

        self.prompts = self._load_prompts()

        try:
            nltk.data.find('tokenizers/punkt')
        except (nltk.downloader.DownloadError, LookupError):
            logging.info("NLTK 'punkt' tokenizer not found. Attempting download...")
            try:
                nltk.download('punkt', quiet=True)
                nltk.data.find('tokenizers/punkt')
                logging.info("'punkt' tokenizer downloaded successfully.")
            except Exception as e:
                logging.error(f"Failed to download NLTK 'punkt' tokenizer: {e}")
                logging.error("Please install it manually: run `python -c \"import nltk; nltk.download('punkt')\"`")
                sys.exit(1)

    def _load_prompts(self) -> Dict[str, str]:
        """Loads system and user prompt templates from files."""
        loaded_prompts = {}
        logger.info(f"Loading prompts from directory: {self.prompts_dir}")
        if not self.prompts_dir.is_dir():
             logger.warning(f"Prompts directory not found: {self.prompts_dir}. Creating it.")
             try:
                  self.prompts_dir.mkdir(parents=True, exist_ok=True)
             except OSError as e:
                  logger.error(f"Could not create prompts directory {self.prompts_dir}: {e}")
                  raise 

        for key, filename in self.PROMPT_FILES.items():
            filepath = self.prompts_dir / filename
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    loaded_prompts[key] = f.read()
                # logger.debug(f"Loaded prompt '{key}' from {filename}")
            except FileNotFoundError:
                logger.error(f"Prompt file not found: {filepath}. Please create it.")
                # raise error if missing
                if key in ['science_system', 'science_user', 'rerank_science_query', 'doi_system', 'doi_user', 'rerank_doi_query']:
                     raise FileNotFoundError(f"Essential prompt file missing: {filepath}")
                else:
                     logger.warning(f"Optional prompt file missing: {filepath}. Validation may not work.")
                     loaded_prompts[key] = "" 

            except Exception as e:
                logger.error(f"Failed to load prompt file {filepath}: {e}")
                raise 

        return loaded_prompts

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        # Modified slightly to include prompts_dir creation check earlier
        for directory in [self.papers_dir, self.texts_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def _load_cache(self, cache_file: Path) -> Dict:
        """Load a cache file if it exists."""
        # Kept user's original logic
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Cache file {cache_file} is corrupted. Starting fresh.")
                return {}
            except Exception as e:
                logging.error(f"Failed to load cache file {cache_file}: {e}")
                return {}
        return {}

    def _save_cache(self, cache_file: Path, data: Dict):
        """Save data to a cache file."""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save cache file {cache_file}: {e}")


    def _get_paper_list(self) -> List[Dict[str, str]]:
        """Fetch list of astronomy papers for the specified month from ADS."""
        # Kept user's original logic, added check for batch mode
        if self.run_mode != "batch" or not self.year_month:
             logger.error("Attempted to get paper list outside of batch mode.")
             return []
        if not self.ads_key:
             logger.error("ADS API key needed for batch mode paper list.")
             return []

        logging.info(f"Fetching paper list for {self.year_month}")

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
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logging.error("ADS API request timed out.")
            return []
        except requests.exceptions.RequestException as e:
            logging.error(f"ADS API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                 logging.error(f"ADS Response Status: {e.response.status_code}")
                 logging.error(f"ADS Response Body: {e.response.text}")
                 if e.response.status_code == 401:
                     logging.error("Check if your ADS_API_KEY is valid.")
            return []

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
            identifiers = doc.get('identifier', [])
            if not isinstance(identifiers, list):
                identifiers = [identifiers]

            arxiv_id = next(
                (id_str.split(':')[1] for id_str in identifiers
                 if isinstance(id_str, str) and id_str.startswith('arXiv:')),
                None
            )
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

    def _is_downloaded_and_converted(self, paper: Dict[str, str]) -> bool:
        """Check if paper PDF is downloaded AND text file exists."""
        downloaded_cache = self._load_cache(self.download_cache)
        txt_path = self.texts_dir / f"{paper['arxiv_id']}.txt"
        pdf_path = self.papers_dir / f"{paper['arxiv_id']}.pdf"

        if self.run_mode == "single":
             return not self.reprocess and pdf_path.exists() and txt_path.exists()

        return (not self.reprocess
                and paper['arxiv_id'] in downloaded_cache
                and pdf_path.exists()
                and txt_path.exists())

    def _is_skipped(self, paper: Dict[str, str]) -> bool:
        """Check if paper was previously skipped."""
        # Kept user's original logic (checks skipped cache)
        # Single mode never checks this cache
        if self.run_mode == "single":
             return False

        skipped = self._load_cache(self.skipped_cache)
        return not self.reprocess and paper['arxiv_id'] in skipped

    # --- Modified: Add save_to_cache flag ---
    def _mark_as_skipped(self, paper: Dict[str, str], reason: str, save_to_cache: bool = True):
        """Mark a paper as skipped with the given reason."""
        arxiv_id = paper['arxiv_id']
        logging.warning(f"Skipping paper {arxiv_id}: {reason}")

        # Only save to cache if flag is True (and implicitly, if in batch mode where cache files make sense)
        if save_to_cache and self.run_mode == "batch":
             skipped = self._load_cache(self.skipped_cache)
             skipped[arxiv_id] = {
                 "reason": reason,
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
             }
             self._save_cache(self.skipped_cache, skipped)

             downloaded = self._load_cache(self.download_cache)
             if arxiv_id in downloaded:
                 del downloaded[arxiv_id]
                 self._save_cache(self.download_cache, downloaded)
        # else: do nothing (don't save skip status for single runs)

    def _download_paper(self, paper: Dict[str, str]) -> bool:
        """
        Download paper PDF from arXiv.
        Returns True if successful, False if paper should be skipped.
        """
        arxiv_id = paper['arxiv_id']
        # Logging adjusted slightly
        logger.info(f"Checking download status for paper {arxiv_id}")

        if self._is_skipped(paper): # Checks cache only in batch mode
             logger.info(f"Paper {arxiv_id} was previously skipped (batch mode check)")
             return False

        pdf_path = self.papers_dir / f"{arxiv_id}.pdf"
        if pdf_path.exists() and not self.reprocess:
             logger.info(f"Paper {arxiv_id} already downloaded.")
             return True

        # Proceed with download attempt
        logger.info(f"Downloading paper {arxiv_id}...")
        # Use user's original URL logic (v1 only)
        url = f"https://arxiv.org/pdf/{arxiv_id}v1"
        # Use user's original headers (if needed, though ADS key likely not needed for PDF download)
        # headers = {"Authorization": f"Bearer {self.ads_key}"} 
        headers = {} 

        try:
            response = requests.get(url, headers=headers, allow_redirects=True)
            response.raise_for_status()

            with open(pdf_path, 'wb') as f:
                f.write(response.content)

            # Update download cache ONLY in batch mode
            if self.run_mode == "batch":
                 downloaded = self._load_cache(self.download_cache)
                 downloaded[paper['arxiv_id']] = True
                 self._save_cache(self.download_cache, downloaded)
            return True

        except requests.exceptions.HTTPError as e:
             if e.response.status_code == 404:
                 logging.warning(f"Paper {arxiv_id} not found (404)")
                 # Mark skipped respecting save_to_cache for current mode
                 self._mark_as_skipped(paper, "404: PDF not found", save_to_cache=(self.run_mode == "batch"))
             else:
                 logging.warning(f"HTTP error downloading {arxiv_id}: {str(e)}")
                 self._mark_as_skipped(paper, f"HTTP error: {str(e)}", save_to_cache=(self.run_mode == "batch"))
             return False

        except Exception as e:
             logging.warning(f"Error downloading {arxiv_id}: {str(e)}")
             self._mark_as_skipped(paper, f"Download error: {str(e)}", save_to_cache=(self.run_mode == "batch"))
             return False


    def _convert_to_text(self, paper: Dict[str, str]) -> bool:
        """Convert PDF to text using pdftext. Returns True if successful, False otherwise."""
        arxiv_id = paper['arxiv_id']
        pdf_path = self.papers_dir / f"{arxiv_id}.pdf"
        txt_path = self.texts_dir / f"{arxiv_id}.txt"

        if txt_path.exists() and not self.reprocess:
            # logger.info(f"Text file for {arxiv_id} already exists.") 
            return True

        if not pdf_path.exists():
            logging.warning(f"Cannot convert {arxiv_id}: PDF file not found at {pdf_path}")
            self._mark_as_skipped(paper, "PDF not found for conversion", save_to_cache=(self.run_mode == "batch"))
            return False

        logging.info(f"Converting paper {arxiv_id} to text")
        try:
            # Use timeout to prevent hangs on problematic PDFs 
            result = subprocess.run(
                ["pdftext", "--sort", str(pdf_path), "--out_path", str(txt_path)],
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=60
            )
            if not txt_path.exists() or txt_path.stat().st_size == 0:
                 logging.error(f"pdftext ran for {arxiv_id}, but output file is missing or empty.")
                 logging.error(f"pdftext stdout: {result.stdout}")
                 logging.error(f"pdftext stderr: {result.stderr}")
                 self._mark_as_skipped(paper, "PDF conversion produced empty/missing file", save_to_cache=(self.run_mode == "batch"))
                 if txt_path.exists(): txt_path.unlink(missing_ok=True) # Use missing_ok
                 return False

            return True

        except subprocess.TimeoutExpired:
             logging.error(f"Timeout converting {arxiv_id} with pdftext.")
             self._mark_as_skipped(paper, "PDF conversion timed out", save_to_cache=(self.run_mode == "batch"))
             if txt_path.exists(): txt_path.unlink(missing_ok=True)
             return False
        except subprocess.CalledProcessError as e:
            logging.error(f"Error converting {arxiv_id} (pdftext exit code {e.returncode}):")
            logging.error(f"  Command: {e.cmd}")
            stderr_decoded = e.stderr # Already text=True in Popen args
            logging.error(f"  Stderr: {stderr_decoded}")
            skip_reason = f"PDF conversion failed (pdftext error): {stderr_decoded[:200]}" # Limit reason length
            self._mark_as_skipped(paper, skip_reason, save_to_cache=(self.run_mode == "batch"))
            if txt_path.exists(): txt_path.unlink(missing_ok=True)
            return False
        except FileNotFoundError:
            logging.error("`pdftext` command not found. Is it installed and in the system PATH?")
            self._mark_as_skipped(paper, "pdftext command not found", save_to_cache=(self.run_mode == "batch"))
            return False
        except Exception as e:
            logging.error(f"Unexpected error converting {arxiv_id}: {str(e)}")
            self._mark_as_skipped(paper, f"Unexpected PDF conversion error: {str(e)}", save_to_cache=(self.run_mode == "batch"))
            if txt_path.exists(): txt_path.unlink(missing_ok=True)
            return False


    def _extract_relevant_snippets(self, paper_text: str, keywords: List[str]) -> List[str]:
        """Extract sentences containing keywords, plus surrounding context sentences."""
        # Kept user's original logic mostly, minor cleanup
        if not paper_text:
            return []

        # preprocess text (user's original logic)
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

        # find indices
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            for keyword in keywords: 
                if keyword in sentence_lower:
                    relevant_indices.add(i)
                    break

        # expand context
        final_indices: Set[int] = set()
        num_sentences = len(sentences) 
        for i in relevant_indices:
            start = max(0, i - self.context_sentences)
            end = min(num_sentences, i + self.context_sentences + 1)
            final_indices.update(range(start, end))

        # create snippets 
        if not final_indices:
            return []

        sorted_indices = sorted(list(final_indices))
        current_snippet_sentences = []
        last_index = -2

        for index in sorted_indices:
            if index != last_index + 1 and current_snippet_sentences:
                extracted_snippets.add(" ".join(current_snippet_sentences).strip())
                current_snippet_sentences = []

            current_snippet_sentences.append(sentences[index])
            last_index = index

        if current_snippet_sentences:
            extracted_snippets.add(" ".join(current_snippet_sentences).strip())

        # logging.debug(f"Extracted {len(extracted_snippets)} unique snippets.")
        return [s for s in list(extracted_snippets) if len(s) > 10]


    def _rerank_snippets(self, query: str, snippets: List[str]) -> List[Dict[str, Union[str, Optional[float]]]]:
        """Rerank snippets using Cohere API based on the query. Returns list of {'snippet': str, 'score': float}."""
        if not self.cohere_client or not snippets:
            return [{'snippet': s, 'score': None} for s in snippets[:self.top_k_snippets]]

        logging.debug(f"Reranking {len(snippets)} snippets with Cohere model '{self.reranker_model}' for query: '{query}'")
        try:
            non_empty_snippets = [s for s in snippets if s and s.strip()]
            if not non_empty_snippets:
                 logging.warning("All extracted snippets were empty after stripping.")
                 return []

            results = self.cohere_client.rerank(
                query=query,
                documents=non_empty_snippets,
                top_n=self.top_k_snippets,
                model=self.reranker_model # Use user's model name
            )

            reranked_data = [{'snippet': non_empty_snippets[result.index], 'score': result.relevance_score} for result in results.results]

            logging.info(f"Reranked scores (top 3): {[f'{r.relevance_score:.3f}' for r in results.results[:3]]}")

            return reranked_data

        except Exception as e:
            logging.error(f"Unexpected error during Cohere reranking: {e}")
            return [{'snippet': s, 'score': None} for s in non_empty_snippets[:self.top_k_snippets]]

    def _needs_analysis(self, paper: Dict[str, str], cache_file: Path) -> bool:
        """Generic check if paper needs analysis based on cache and reprocess flag."""
        if self.run_mode == 'single':
            return True
        analyzed = self._load_cache(cache_file)
        return self.reprocess or paper['arxiv_id'] not in analyzed

    def _call_openai_parse(self, model_name: str, system_prompt: str, user_prompt: str, response_model: type[BaseModel]) -> Optional[Dict[str, Any]]: 
        """Helper function to call OpenAI parse API with error handling."""
        try:
            if not system_prompt or not user_prompt:
                logger.error("System or User prompt is empty. Cannot call OpenAI.")
                return {"error": "empty_prompt", "message": "System or User prompt was empty."}

            result = self.openai_client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=response_model,
                timeout=60 
            )

            parsed_object = result.choices[0].message.parsed
            if parsed_object:
                if isinstance(parsed_object, response_model):
                    return parsed_object.model_dump()
                else:
                    logger.error(f"OpenAI response parsed, but yielded unexpected type: {type(parsed_object)}. Expected {response_model}.")
                    logger.debug(f"Raw parsed object: {parsed_object}")
                    return {"error": "parse_type_mismatch", "message": f"Parsed object type mismatch: got {type(parsed_object)}"}
            else:
                logging.error("OpenAI response parsed, but no Pydantic object found in expected location.")
                logging.debug(f"Full OpenAI response object: {result}")
                try:
                    raw_content = result.choices[0].message.content
                    if isinstance(raw_content, str):
                        parsed_json = json.loads(raw_content)
                        validated_model = response_model.model_validate(parsed_json)
                        logger.warning("Used manual JSON parsing/validation fallback.")
                        return validated_model.model_dump()
                    else:
                        logging.error("Raw message content is not a string for manual parsing.")
                        return None
                except (json.JSONDecodeError, ValidationError, Exception) as manual_parse_err:
                    logging.error(f"Manual parsing/validation of OpenAI response failed: {manual_parse_err}")
                    return None

        except BadRequestError as e:
            logging.warning(f"OpenAI API BadRequestError: {e}")
            if "context length" in str(e).lower():
                logging.warning(f"Snippets might still exceed token limit: {e}")
                return {"error": "token_limit", "message": str(e)}
            else:
                return {"error": "bad_request", "message": str(e)}

        except Exception as e:
            logging.error(f"OpenAI API call failed: {e}")
            return {"error": "api_error", "message": str(e)}

        return None 

    def _analyze_science(self, paper: Dict[str, str], save_to_cache: bool = True) -> Dict:
        """Analyze paper for JWST science content using extracted snippets."""
        arxiv_id = paper['arxiv_id']
        logger.info(f"Analyzing JWST science content for {arxiv_id}")
        science_results = self._load_cache(self.science_cache) if save_to_cache else {}

        txt_path = self.texts_dir / f"{arxiv_id}.txt"
        if not txt_path.exists():
             logger.warning(f"Text file not found for {arxiv_id}, cannot analyze science.")
             return {"jwstscience": -1.0, "reason": "Analysis failed: Text file missing", "quotes": [], "error": "missing_text_file"}

        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                paper_text = f.read()
        except Exception as e:
            logger.error(f"Failed to read text file {txt_path}: {e}")
            return {"jwstscience": -1.0, "reason": "Analysis failed: Cannot read text file", "quotes": [], "error": "read_error"}

        # 1. Extract Snippets (using user's original keywords)
        keywords_to_find = self.SCIENCE_KEYWORDS_LOWER
        all_snippets = self._extract_relevant_snippets(paper_text, keywords_to_find)

        if not all_snippets:
            logger.info(f"No relevant keywords found for science analysis in {arxiv_id}.")
            result = {"jwstscience": 0.0, "quotes": [], "reason": "No relevant keywords (e.g., JWST, instruments) found in text."}
            if save_to_cache and self.run_mode == "batch":
                science_results[arxiv_id] = result
                self._save_cache(self.science_cache, science_results)
            return result

        # 2. Rerank Snippets 
        rerank_query = self.prompts.get('rerank_science_query')
        if not rerank_query:
            logger.error("Rerank science query prompt ('rerank_science_query.txt') not found or empty.")
            return {"jwstscience": -1.0, "reason": "Analysis failed: Missing rerank science query prompt", "quotes": [], "error": "prompt_missing"}

        reranked_data = self._rerank_snippets(rerank_query, all_snippets)

        if not reranked_data:
            logger.warning(f"Reranking produced no snippets for {arxiv_id}. Skipping LLM analysis.")
            result = {"jwstscience": 0.0, "quotes": [], "reason": "Keyword snippets found but none survived reranking/filtering."}
            if save_to_cache and self.run_mode == "batch":
                science_results[arxiv_id] = result
                self._save_cache(self.science_cache, science_results)
            return result
        elif top_score := reranked_data[0].get('score') < self.reranker_threshold:
            logger.info(f"Skipping LLM science analysis for {arxiv_id}: Top reranker score ({top_score:g}) below threshold ({self.reranker_threshold}).")
            result = {
                "jwstscience": 0.0, 
                "quotes": [],
                "reason": f"Skipped LLM analysis: Top reranker score ({top_score:g}) was below the threshold ({self.reranker_threshold}).",
            }
            if save_to_cache and self.run_mode == "batch":
                science_results = self._load_cache(self.science_cache) # Reload cache
                science_results[arxiv_id] = result
                self._save_cache(self.science_cache, science_results)
            return result 

        if save_to_cache and self.run_mode == "batch":
             try:
                snippets_cache_data = self._load_cache(self.snippets_cache)
                if arxiv_id not in snippets_cache_data: snippets_cache_data[arxiv_id] = {}
                snippets_cache_data[arxiv_id]["science_analysis"] = reranked_data 
                self._save_cache(self.snippets_cache, snippets_cache_data)
             except Exception as e:
                logging.error(f"Failed to save science snippets cache for {arxiv_id}: {e}")

        # 3. Prepare LLM Input (user's original truncation logic)
        reranked_snippets_for_llm = [item['snippet'] for item in reranked_data]
        snippets_text = "\n---\n".join([f"Excerpt {i+1}:\n{s}" for i, s in enumerate(reranked_snippets_for_llm)])
        max_chars = 30000 
        if len(snippets_text) > max_chars:
            logger.warning(f"Total snippet text for {arxiv_id} exceeds {max_chars} chars, truncating.")
            snippets_text = snippets_text[:max_chars]

        system_prompt = self.prompts.get('science_system')
        user_prompt_template = self.prompts.get('science_user')
        if not system_prompt or not user_prompt_template:
            logger.error(f"Science prompts not loaded correctly for {arxiv_id}. Check prompts directory.")
            return {"jwstscience": -1.0, "reason": "Analysis failed: Prompts missing", "quotes": [], "error": "prompt_missing"}
        try:
            user_prompt = user_prompt_template.format(snippets_text=snippets_text)
        except KeyError:
            logger.error("Failed to format science user prompt - missing '{snippets_text}' placeholder?")
            return {"jwstscience": -1.0, "reason": "Analysis failed: Prompt formatting error", "quotes": [], "error": "prompt_format_error"}

        # 4. Call LLM (Initial Analysis)
        llm_result = self._call_openai_parse(self.gpt_model, system_prompt, user_prompt, JWSTScienceLabelerModel)

        if llm_result is None or "error" in llm_result:
            error_reason = f"LLM analysis failed: {llm_result.get('message', 'Unknown error') if llm_result else 'Unknown error'}"
            error_type = llm_result.get('error', 'unknown') if llm_result else 'unknown'
            self._mark_as_skipped(paper, error_reason, save_to_cache=save_to_cache)
            return {"jwstscience": -1.0, "reason": error_reason, "quotes": [], "error": error_type}

        final_result = llm_result

        # 5. optional Validation Step 
        if self.validate_llm and final_result.get("jwstscience", 0.0) > 0:
            logging.debug(f"Performing LLM validation for science score on {arxiv_id}")
            validation_system_prompt = self.prompts.get('science_validate_system')
            validation_user_template = self.prompts.get('science_validate_user')
            if not validation_system_prompt or not validation_user_template:
                 logger.error(f"Science validation prompts not loaded correctly for {arxiv_id}.")
                 validation_llm_result = {"error": "prompt_missing", "message":"Validation prompts missing"}
            else:
                try:
                    # format validation prompt
                    validation_user_prompt = validation_user_template.format(
                        original_quotes=json.dumps(final_result.get("quotes", [])),
                        original_score=final_result.get("jwstscience", "N/A"),
                        original_reason=final_result.get("reason", "N/A"),
                    )
                except KeyError as e:
                    logger.error(f"Failed to format science validation user prompt - missing placeholder {e}?")
                    validation_llm_result = {"error": "prompt_format_error", "message": f"Validation prompt format error: {e}"}
                else:
                      validation_llm_result = self._call_openai_parse(self.gpt_model, validation_system_prompt, validation_user_prompt, JWSTScienceLabelerModel)

            # if validation_llm_result and "error" not in validation_llm_result:
            #     if validation_llm_result.get("quotes") == final_result.get("quotes"):
            #           final_result = validation_llm_result
            #     else:
            #           logging.warning(f"LLM validation for {arxiv_id} changed quotes. Discarding validation.")
            # elif validation_llm_result:
            #     logging.warning(f"LLM validation step failed for {arxiv_id}: {validation_llm_result.get('message', 'Unknown error')}")


        # 6. Cache the final result (conditional)
        if save_to_cache and self.run_mode == "batch":
            science_results = self._load_cache(self.science_cache) # Reload cache
            science_results[arxiv_id] = final_result
            self._save_cache(self.science_cache, science_results)

        return final_result

    def _analyze_doi(self, paper: Dict[str, str], save_to_cache: bool = True) -> Dict:
        """Analyze paper for JWST DOIs using extracted snippets."""
        arxiv_id = paper['arxiv_id']
        logging.info(f"Analyzing JWST DOIs for {arxiv_id}")
        doi_results = self._load_cache(self.doi_cache) if save_to_cache else {}

        txt_path = self.texts_dir / f"{arxiv_id}.txt"
        if not txt_path.exists():
            logger.warning(f"Text file not found for {arxiv_id}, cannot analyze DOI.")
            return {"jwstdoi": -1.0, "reason": "Analysis failed: Text file missing", "quotes": [], "error": "missing_text_file"}

        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                paper_text = f.read()
        except Exception as e:
            logging.error(f"Failed to read text file {txt_path}: {e}")
            return {"jwstdoi": -1.0, "reason": "Analysis failed: Cannot read text file", "quotes": [], "error": "read_error"}

        # 1. Extract Snippets (using user's original keyword logic)
        keywords_to_find = self.DOI_KEYWORDS_LOWER + [k for k in self.SCIENCE_KEYWORDS_LOWER if k not in self.DOI_KEYWORDS_LOWER] # Combine
        all_snippets = self._extract_relevant_snippets(paper_text, keywords_to_find)

        if not all_snippets:
            logging.info(f"No relevant keywords found for DOI analysis in {arxiv_id}.")
            result = {"jwstdoi": 0.0, "quotes": [], "reason": "No relevant keywords (e.g., DOI, 10.17909, data availability, JWST) found in text."}
            if save_to_cache and self.run_mode == "batch":
                doi_results[arxiv_id] = result
                self._save_cache(self.doi_cache, doi_results)
            return result

        # 2. Rerank Snippets 
        rerank_query = self.prompts.get('rerank_doi_query')
        if not rerank_query:
            logger.error("Rerank DOI query prompt ('rerank_doi_query.txt') not found or empty.")
            return {"jwstdoi": -1.0, "reason": "Analysis failed: Missing rerank DOI query prompt", "quotes": [], "error": "prompt_missing"}
        reranked_data = self._rerank_snippets(rerank_query, all_snippets)

        if not reranked_data:
            logging.warning(f"Reranking produced no snippets for DOI analysis {arxiv_id}. Skipping LLM.")
            result = {"jwstdoi": 0.0, "quotes": [], "reason": "Keyword snippets found but none survived reranking/filtering for DOI check."}
            if save_to_cache and self.run_mode == "batch":
                doi_results[arxiv_id] = result
                self._save_cache(self.doi_cache, doi_results)
            return result

        if save_to_cache and self.run_mode == "batch":
            try:
                snippets_cache_data = self._load_cache(self.snippets_cache)
                if arxiv_id not in snippets_cache_data: snippets_cache_data[arxiv_id] = {}
                snippets_cache_data[arxiv_id]["doi_analysis"] = reranked_data 
                self._save_cache(self.snippets_cache, snippets_cache_data)
            except Exception as e:
                logging.error(f"Failed to save DOI snippets cache for {arxiv_id}: {e}")


        # 3. Prepare LLM Input
        reranked_snippets_for_llm = [item['snippet'] for item in reranked_data]
        snippets_text = "\n---\n".join([f"Excerpt {i+1}:\n{s}" for i, s in enumerate(reranked_snippets_for_llm)])
        max_chars = 30000 
        if len(snippets_text) > max_chars:
            logging.warning(f"Total snippet text for DOI analysis {arxiv_id} exceeds {max_chars} chars, truncating.")
            snippets_text = snippets_text[:max_chars]

        system_prompt = self.prompts.get('doi_system')
        user_prompt_template = self.prompts.get('doi_user')
        if not system_prompt or not user_prompt_template:
            logger.error(f"DOI prompts not loaded correctly for {arxiv_id}. Check prompts directory.")
            return {"jwstdoi": -1.0, "reason": "Analysis failed: Prompts missing", "quotes": [], "error": "prompt_missing"}
        try:
            user_prompt = user_prompt_template.format(snippets_text=snippets_text)
        except KeyError:
            logger.error("Failed to format DOI user prompt - missing '{snippets_text}' placeholder?")
            return {"jwstdoi": -1.0, "reason": "Analysis failed: Prompt formatting error", "quotes": [], "error": "prompt_format_error"}

        # 4. Call LLM (Initial Analysis)
        llm_result = self._call_openai_parse(self.gpt_model, system_prompt, user_prompt, JWSTDOILabelerModel)

        if llm_result is None or "error" in llm_result:
            error_reason = f"LLM analysis failed for DOI: {llm_result.get('message', 'Unknown error') if llm_result else 'Unknown error'}"
            error_type = llm_result.get('error', 'unknown') if llm_result else 'unknown'
            logging.warning(f"DOI analysis failed for {arxiv_id}: {error_reason}")
            return {"jwstdoi": -1.0, "reason": error_reason, "quotes": [], "error": error_type}

        final_result = llm_result

        # 5. Optional Validation Step 
        if self.validate_llm and final_result.get("jwstdoi", 0.0) > 0:
            logging.debug(f"Performing LLM validation for DOI score on {arxiv_id}")
            validation_system_prompt = self.prompts.get('doi_validate_system')
            validation_user_template = self.prompts.get('doi_validate_user')
            if not validation_system_prompt or not validation_user_template:
                 logger.error(f"DOI validation prompts not loaded correctly for {arxiv_id}.")
                 validation_llm_result = {"error": "prompt_missing", "message": "Validation prompts missing"}
            else:
                try:
                    validation_user_prompt = validation_user_template.format(
                        original_quotes=json.dumps(final_result.get("quotes", []), indent=2),
                        original_score=final_result.get("jwstdoi", "N/A"),
                        original_reason=final_result.get("reason", "N/A"),
                    )
                except KeyError as e:
                    logger.error(f"Failed to format DOI validation user prompt - missing placeholder {e}?")
                    validation_llm_result = {"error": "prompt_format_error", "message": f"Validation prompt format error: {e}"}
                else:
                    validation_llm_result = self._call_openai_parse(self.gpt_model, validation_system_prompt, validation_user_prompt, JWSTDOILabelerModel)

            # if validation_llm_result and "error" not in validation_llm_result:
            #     if validation_llm_result.get("quotes") == final_result.get("quotes"):
            #           final_result = validation_llm_result
            #     else:
            #           logging.warning(f"LLM DOI validation for {arxiv_id} changed quotes. Discarding validation.")
            # elif validation_llm_result:
            #      logging.warning(f"LLM DOI validation step failed for {arxiv_id}: {validation_llm_result.get('message', 'Unknown error')}")

        # 6. Cache the final result (conditional)
        if save_to_cache and self.run_mode == "batch":
            doi_results = self._load_cache(self.doi_cache) 
            doi_results[arxiv_id] = final_result
            self._save_cache(self.doi_cache, doi_results)

        return final_result

    def _generate_report(self):
        """Generate a summary report of the analysis."""
        if self.run_mode != "batch":
             logger.error("Attempted to generate report outside of batch mode.")
             return None

        science_results = self._load_cache(self.science_cache)
        doi_results = self._load_cache(self.doi_cache)
        skipped_results = self._load_cache(self.skipped_cache)
        downloaded_papers = self._load_cache(self.download_cache) # User's original way to count attempted

        total_attempted = len(downloaded_papers) + len(skipped_results)
        successfully_processed = {k: v for k, v in science_results.items() if isinstance(v, dict) and v.get("jwstscience", -1.0) >= 0}
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

        # Loop through science papers as per original logic
        for arxiv_id in science_papers_ids:
             doi_info = doi_results.get(arxiv_id)
             science_info = successfully_processed[arxiv_id] # Should exist
             doi_score = 0.0
             doi_reason = "DOI analysis not available or failed"
             doi_quotes: List[str] = []
             has_valid_doi = False 

             if doi_info and isinstance(doi_info, dict) and doi_info.get("jwstdoi", -1.0) >= 0: # Check structure
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


        report = {
            "metadata": {
                "report_generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "year_month_analyzed": self.year_month,
                "science_threshold": self.science_threshold,
                "doi_threshold": self.doi_threshold,
                "gpt_model": self.gpt_model,
                "reranker_model": self.reranker_model if self.cohere_client else "N/A (Cohere unavailable)",
                "top_k_snippets": self.top_k_snippets,
                "context_sentences": self.context_sentences,
                "llm_validation_enabled": self.validate_llm,
                "prompts_directory": str(self.prompts_dir.resolve()),
            },
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
            "jwst_science_papers_details": sorted(detailed_results_list, key=lambda x: x['arxiv_id']) # Sort by ID
        }

        report_path = self.results_dir / f"{self.year_month}_report.json"
        try: # Wrap saving in try-except
             with open(report_path, 'w', encoding='utf-8') as f:
                 json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
             logging.error(f"Failed to save report file {report_path}: {e}")
             # Don't crash the whole run if report saving fails

        logging.info(f"Report generated: {report_path}")
        # --- Print Summary to Log (user's original fields) ---
        logging.info(f"--- Summary for {self.year_month} ---")
        logging.info(f"  Total papers from ADS: {report['summary']['total_papers_identified_from_ads']}")
        logging.info(f"  Skipped (Download/Convert): {report['summary']['papers_skipped_before_analysis']}")
        logging.info(f"  Analysis Failed (LLM/etc): {report['summary']['papers_analysis_failed']}")
        logging.info(f"  Successfully Analyzed: {report['summary']['papers_successfully_analyzed']}")
        logging.info(f"  -> JWST Science Papers (Score >= {self.science_threshold}): {science_papers_count}")
        logging.info(f"     -> With Valid DOI (Score >= {self.doi_threshold}): {papers_with_dois_count}")
        logging.info(f"     -> Missing Valid DOI: {papers_missing_dois_count}")
        logging.info("--- End Summary ---")

        return report

    def run_batch(self):
        """Main execution pipeline for batch mode"""
        if self.run_mode != "batch":
             logger.error("run_batch called in non-batch mode.")
             return

        start_time = time.time()
        logger.info(f"Starting analysis for {self.year_month}...")

        try:
            # 1. get paper list from ADS
            papers = self._get_paper_list()
            if not papers:
                 logging.warning("No papers found or ADS query failed. Exiting.")
                 self._generate_report() 
                 return

            # 2. process each paper: Download -> Convert -> Analyze
            processed_ids = set() 

            for i, paper in enumerate(papers):
                 arxiv_id = paper['arxiv_id']
                 logging.info(f"Processing paper {i+1}/{len(papers)}: {arxiv_id}")

                 if self._is_skipped(paper):
                     logging.info(f"Paper {arxiv_id} was previously skipped. Skipping.")
                     continue

                 # 2a. download PDF (respects reprocess, uses cache)
                 if not self._download_paper(paper): 
                      continue 

                 # 2b. convert PDF to Text (respects reprocess, uses cache)
                 if not self._convert_to_text(paper): 
                      continue

                 processed_ids.add(arxiv_id) 

                 # 3. analyze for JWST Science
                 science_result = None
                 if self._needs_analysis(paper, self.science_cache):
                     science_result = self._analyze_science(paper, save_to_cache=True)
                 else:
                     # load from cache if not reprocessing
                     science_result = self._load_cache(self.science_cache).get(arxiv_id)
                     if science_result and isinstance(science_result, dict): # Add type check
                          logging.info(f"Using cached science analysis for {arxiv_id}")
                     else:
                          logging.warning(f"Cache logic error: Science cache missing or invalid for {arxiv_id}. Re-analyzing.")
                          science_result = self._analyze_science(paper, save_to_cache=True)

                 # Use score from result, check for error structure
                 current_science_score = -1.0
                 if science_result and isinstance(science_result, dict) and "error" not in science_result:
                      current_science_score = science_result.get("jwstscience", -1.0)
                 else:
                      logger.warning(f"Science analysis failed for {arxiv_id}, cannot proceed to DOI analysis.")
                      # Should already be marked skipped by _analyze_science if needed
                      continue # Skip DOI analysis if science failed


                 if current_science_score < self.science_threshold:
                     logging.info(f"Paper {arxiv_id} does not meet science threshold ({current_science_score:.2f} < {self.science_threshold}). Skipping DOI analysis.")
                     # doi_results = self._load_cache(self.doi_cache)
                     # if arxiv_id in doi_results: del doi_results[arxiv_id] # Clear old result maybe?
                     # self._save_cache...
                     continue

                 # 4. Analyze for DOIs (only if it's a science paper)
                 if self._needs_analysis(paper, self.doi_cache):
                      doi_result = self._analyze_doi(paper, save_to_cache=True)
                      # Check if DOI analysis failed
                      if doi_result is None or doi_result.get("jwstdoi", -1.0) < 0:
                            logger.warning(f"DOI analysis failed for science paper {arxiv_id}.")
                            # Don't skip paper, report will reflect failed DOI analysis
                 else:
                      doi_result_cached = self._load_cache(self.doi_cache).get(arxiv_id)
                      if doi_result_cached and isinstance(doi_result_cached, dict): # Add type check
                           logging.info(f"Using cached DOI analysis for {arxiv_id}")
                      else:
                           # This case is possible if DOI analysis failed previously or was skipped
                           logging.warning(f"DOI analysis needed for {arxiv_id} but cache missing/incomplete. Re-analyzing.")
                           doi_result = self._analyze_doi(paper, save_to_cache=True)
                           if doi_result is None or doi_result.get("jwstdoi", -1.0) < 0:
                                 logger.warning(f"DOI re-analysis also failed for science paper {arxiv_id}.")

                 # time.sleep(1)


            # 5. Generate final summary report
            report = self._generate_report()

            end_time = time.time()
            logger.info(f"Analysis complete for {self.year_month} in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            logging.exception(f"An unhandled error occurred during the run: {e}") # Log stack trace
            try:
                logging.info("Attempting to generate partial report after error...")
                self._generate_report()
            except Exception as report_err:
                logging.error(f"Failed to generate partial report: {report_err}")
            raise 


    def process_single_paper(self, arxiv_id: str):
        """Processes a single paper by arXiv ID and prints results to stdout."""
        start_time = time.time()
        logger.info(f"Starting SINGLE analysis for arXiv ID: {arxiv_id}")

        # Basic validation (can be done in main too)
        if not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", arxiv_id.split('/')[-1]):
            logger.error(f"Invalid arXiv ID format provided: {arxiv_id}")
            print(json.dumps({"error": "invalid_arxiv_id", "arxiv_id": arxiv_id, "message": "Invalid arXiv ID format."}, indent=2))
            return

        # Create a dummy paper dictionary
        paper = {'arxiv_id': arxiv_id, 'bibcode': 'N/A'} 

        final_output = {
            "arxiv_id": arxiv_id,
            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
            "processed_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "status": "Started",
            "science_analysis": None,
            "doi_analysis": None,
            "error_info": None
        }

        try:
            if not self._download_paper(paper): 
                logger.error(f"Failed to download PDF for {arxiv_id}.")
                final_output["status"] = "Error: Download Failed"
                final_output["error_info"] = "Could not retrieve PDF from arXiv (check logs for details)."
                # Print and exit
                print(json.dumps(final_output, indent=2, ensure_ascii=False))
                return

            if not self._convert_to_text(paper): 
                logger.error(f"Failed to convert PDF to text for {arxiv_id}.")
                final_output["status"] = "Error: Conversion Failed"
                final_output["error_info"] = "pdftext tool failed or produced empty output (check logs)."
                # Print and exit
                print(json.dumps(final_output, indent=2, ensure_ascii=False))
                return

            logger.info(f"Analyzing science content for single paper {arxiv_id}")
            science_result = self._analyze_science(paper, save_to_cache=False)
            final_output["science_analysis"] = science_result 

            current_science_score = -1.0
            if science_result and isinstance(science_result, dict) and "error" not in science_result:
                 current_science_score = science_result.get("jwstscience", -1.0)
                 final_output["status"] = "Science Analysis Complete" 
            else:
                 logger.error(f"Science analysis failed for {arxiv_id}. See results.")
                 final_output["status"] = "Error: Science Analysis Failed"
                 final_output["error_info"] = science_result.get("reason", "Science analysis failed or returned invalid data") if science_result else "Science analysis failed"
                 print(json.dumps(final_output, indent=2, ensure_ascii=False))
                 return

            if current_science_score >= self.science_threshold:
                logger.info(f"Science score >= threshold. Analyzing DOI for {arxiv_id}")
                doi_result = self._analyze_doi(paper, save_to_cache=False)
                final_output["doi_analysis"] = doi_result # Store result

                # Check if DOI analysis failed
                if doi_result and isinstance(doi_result, dict) and "error" not in doi_result:
                     final_output["status"] = "Complete" # Both ran ok
                else:
                     logger.error(f"DOI analysis failed for {arxiv_id}. See results.")
                     final_output["status"] = "Complete (with DOI Analysis Error)"
            else:
                logger.info(f"Science score below threshold. Skipping DOI analysis for {arxiv_id}.")
                final_output["status"] = "Complete (DOI Skipped - Low Science Score)"
                final_output["doi_analysis"] = {"jwstdoi": 0.0, "reason": "Skipped due to low science score", "quotes": [], "error": None}


        except Exception as e:
            logger.exception(f"Unhandled error during single paper processing for {arxiv_id}: {e}")
            final_output["status"] = "Error: Unhandled Exception"
            final_output["error_info"] = f"Unexpected error: {str(e)}"

        try:
            # Use ensure_ascii=False for potentially non-ascii quotes
            print(json.dumps(final_output, indent=2, ensure_ascii=False))
        except Exception as json_e:
            logger.error(f"Failed to serialize final result to JSON for {arxiv_id}: {json_e}")
            # Print basic error if JSON fails
            print(f'{{"error": "json_serialization_failed", "arxiv_id": "{arxiv_id}", "message": "{str(json_e)}"}}')

        end_time = time.time()
        logger.info(f"Single analysis for {arxiv_id} finished in {end_time - start_time:.2f} seconds. Status: {final_output['status']}")


def main():
    parser = argparse.ArgumentParser(
        description="Download arXiv papers via ADS, extract text, rerank snippets, and use LLMs to classify JWST science content and DOI presence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--year-month", 
        help="Month to analyze in YYYY-MM format (e.g., 2024-01) for batch processing."
    )
    mode_group.add_argument(
        "--arxiv-id", 
        help="Specific arXiv ID (e.g., 2301.12345) to process for single paper analysis."
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./"), # User's original default
        help="Directory to store all outputs (papers, texts, results)"
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
        default=0.05, # set to 0.0 to turn off by default 
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
        "--reranker-model",
        default="rerank-v3.5", 
        help="Cohere reranker model name"
    )
    parser.add_argument(
        "--gpt-model",
        default="gpt-4.1-mini-2025-04-14", # "gpt-4o-mini-2024-07-18", 
        help="GPT scoring model for JWST science and DOIs"
    )
    parser.add_argument(
        "--validate-llm",
        action="store_true",
        help="Perform a second LLM call to validate the first analysis (increases cost/time)"
    )
    parser.add_argument("--ads-key", help="ADS API key (uses ADS_API_KEY env var if not provided)")
    parser.add_argument("--openai-key", help="OpenAI API key (uses OPENAI_API_KEY env var if not provided)")
    parser.add_argument("--cohere-key", help="Cohere API key (uses COHERE_API_KEY env var if not provided; reranking skipped if missing)")


    args = parser.parse_args()

    # Validate thresholds (user's original checks)
    if not 0 <= args.science_threshold <= 1:
        parser.error("Science threshold must be between 0 and 1")
    if not 0 <= args.doi_threshold <= 1:
        parser.error("DOI threshold must be between 0 and 1")

    # Validate formats superficially (user's original checks + new one)
    if args.year_month and not re.match(r"^\d{4}-\d{2}$", args.year_month):
         parser.error("year_month format must be YYYY-MM")
    if args.arxiv_id and not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", args.arxiv_id.split('/')[-1]):
         parser.error("Invalid arXiv ID format. Should be like XXXX.YYYYY or XXXX.YYYYYvN")


    # Create output/prompts directory if it doesn't exist
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.prompts_dir.mkdir(parents=True, exist_ok=True) # Also ensure prompts dir exists
    except Exception as e:
        logging.error(f"Failed to create necessary directories ({args.output_dir}, {args.prompts_dir}): {e}")
        sys.exit(1)


    try:
        analyzer = JWSTPreprintDOIAnalyzer(
            # Pass necessary args based on mode
            year_month=args.year_month, # Will be None if arxiv_id provided
            arxiv_id=args.arxiv_id,     # Will be None if year_month provided
            output_dir=args.output_dir,
            prompts_dir=args.prompts_dir, 
            science_threshold=args.science_threshold,
            doi_threshold=args.doi_threshold,
            reranker_threshold=args.reranker_threshold,
            ads_key=args.ads_key,
            openai_key=args.openai_key,
            cohere_key=args.cohere_key,
            gpt_model=args.gpt_model,
            reranker_model=args.reranker_model,
            top_k_snippets=args.top_k_snippets,
            context_sentences=args.context_sentences,
            validate_llm=args.validate_llm,
            reprocess=args.reprocess,
        )

        if analyzer.run_mode == "batch":
             analyzer.run_batch() 
        elif analyzer.run_mode == "single":
             analyzer.process_single_paper(args.arxiv_id) 

    except ValueError as e:
        logging.error(f"Initialization Error: {e}")
        sys.exit(1)
    except FileNotFoundError as e: # Catch missing prompt files
         logging.error(f"Setup Error: {e}")
         sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during execution: {e}")
        logging.exception("Traceback:")
        sys.exit(1)

    logging.info("Script finished successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()