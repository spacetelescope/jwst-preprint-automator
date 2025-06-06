"""Cache management utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_cache(cache_file: Path) -> Dict[str, Any]:
    """Load a cache file if it exists."""
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Cache file {cache_file} is corrupted. Starting fresh.")
            return {}
        except Exception as e:
            logger.error(f"Failed to load cache file {cache_file}: {e}")
            return {}
    return {}


def save_cache(cache_file: Path, data: Dict[str, Any]) -> None:
    """Save data to a cache file."""
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save cache file {cache_file}: {e}")