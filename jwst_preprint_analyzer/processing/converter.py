"""PDF to text conversion functionality."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class PDFConverter:
    """Handles conversion of PDF files to text."""
    
    def __init__(self, texts_dir: Path):
        self.texts_dir = texts_dir
        
    def convert_to_text(self, arxiv_id: str, pdf_path: Path, reprocess: bool = False) -> bool:
        """Convert PDF to text using pdftext. Returns True if successful, False otherwise."""
        txt_path = self.texts_dir / f"{arxiv_id}.txt"

        if txt_path.exists() and not reprocess:
            return True

        if not pdf_path.exists():
            logger.warning(f"Cannot convert {arxiv_id}: PDF file not found at {pdf_path}")
            return False

        logger.info(f"Converting paper {arxiv_id} to text")
        try:
            result = subprocess.run(
                ["pdftext", "--sort", str(pdf_path), "--out_path", str(txt_path)],
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=60
            )
            if not txt_path.exists() or txt_path.stat().st_size == 0:
                logger.error(f"pdftext ran for {arxiv_id}, but output file is missing or empty.")
                logger.error(f"pdftext stdout: {result.stdout}")
                logger.error(f"pdftext stderr: {result.stderr}")
                if txt_path.exists(): 
                    txt_path.unlink(missing_ok=True)
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.error(f"Timeout converting {arxiv_id} with pdftext.")
            if txt_path.exists(): 
                txt_path.unlink(missing_ok=True)
            return False
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting {arxiv_id} (pdftext exit code {e.returncode}):")
            logger.error(f"  Command: {e.cmd}")
            stderr_decoded = e.stderr
            logger.error(f"  Stderr: {stderr_decoded}")
            if txt_path.exists(): 
                txt_path.unlink(missing_ok=True)
            return False
            
        except FileNotFoundError:
            logger.error("`pdftext` command not found. Is it installed and in the system PATH?")
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error converting {arxiv_id}: {str(e)}")
            if txt_path.exists(): 
                txt_path.unlink(missing_ok=True)
            return False