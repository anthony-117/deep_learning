import re
import time
from datetime import datetime
from threading import Lock
from typing import List, Optional, Dict, Any
from pathlib import Path
from functools import wraps

from .base import PaperMetadata
from .exceptions import RateLimitError


class RateLimiter:
    def __init__(self, requests_per_period: int, period_seconds: int):
        self.requests_per_period = requests_per_period
        self.period_seconds = period_seconds
        self.tokens = requests_per_period
        self.last_update = time.time()
        self.lock = Lock()

    def _add_tokens(self):
        now = time.time()
        elapsed = now - self.last_update

        if elapsed > 0:
            # Calculate tokens to add
            tokens_to_add = (elapsed / self.period_seconds) * self.requests_per_period
            self.tokens = min(self.requests_per_period, self.tokens + tokens_to_add)
            self.last_update = now

    def acquire(self, source: str = "unknown", blocking: bool = True) -> bool:
        with self.lock:
            self._add_tokens()

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            else:
                if blocking:
                    # Wait until next token available
                    wait_time = self.period_seconds / self.requests_per_period
                    time.sleep(wait_time)
                    self.tokens = 0  # Token was used during wait
                    return True
                else:
                    # Calculate retry after time
                    retry_after = (1 - self.tokens) * (self.period_seconds / self.requests_per_period)
                    raise RateLimitError(source, retry_after)

    def __repr__(self) -> str:
        return f"RateLimiter(tokens={self.tokens:.2f}/{self.requests_per_period}, period={self.period_seconds}s)"


class SimpleCache:
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                else:
                    # Expired, remove from cache
                    del self.cache[key]
            return None

    def set(self, key: str, value: Any):
        with self.lock:
            self.cache[key] = (value, time.time())

    def clear(self):
        """Clear all cached values."""
        with self.lock:
            self.cache.clear()

    def __len__(self) -> int:
        return len(self.cache)


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)

    # Replace spaces and other whitespace with underscores
    filename = re.sub(r'\s+', '_', filename)

    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')

    # Truncate if too long
    if len(filename) > max_length:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        if ext:
            name = name[:max_length - len(ext) - 1]
            filename = f"{name}.{ext}"
        else:
            filename = filename[:max_length]

    return filename


def normalize_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None

    # Common date formats
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%d %b %Y",
        "%B %d, %Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def deduplicate_papers(
    papers: List[PaperMetadata],
    method: str = "doi"
) -> List[PaperMetadata]:
    """
    Args:
        papers: List of PaperMetadata objects
        method: Deduplication method ("doi", "title", or "strict")
            - "doi": Match by DOI (if available)
            - "title": Fuzzy match by title similarity
            - "strict": DOI match only, keep all papers without DOI
    """
    if not papers:
        return []

    seen = set()
    unique_papers = []

    for paper in papers:
        if method == "doi":
            # Deduplicate by DOI if available
            if paper.doi:
                if paper.doi not in seen:
                    seen.add(paper.doi)
                    unique_papers.append(paper)
            else:
                # No DOI, keep the paper
                unique_papers.append(paper)

        elif method == "title":
            # Normalize title for comparison
            normalized_title = paper.title.lower().strip()
            normalized_title = re.sub(r'\s+', ' ', normalized_title)

            if normalized_title not in seen:
                seen.add(normalized_title)
                unique_papers.append(paper)

        elif method == "strict":
            # Only deduplicate papers with DOI
            if paper.doi:
                if paper.doi not in seen:
                    seen.add(paper.doi)
                    unique_papers.append(paper)
            else:
                unique_papers.append(paper)

    return unique_papers


def format_authors(authors: List[Any], max_authors: int = 3) -> str:
    if not authors:
        return "Unknown"

    # Extract names
    names = []
    for author in authors:
        if hasattr(author, 'name'):
            names.append(author.name)
        else:
            names.append(str(author))

    # Format based on count
    if len(names) <= max_authors:
        return ", ".join(names)
    else:
        displayed = ", ".join(names[:max_authors])
        return f"{displayed} et al. ({len(names)} total)"


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(current_delay)
                    current_delay *= backoff
            return None
        return wrapper
    return decorator


def create_output_directory(base_dir: str, source: str = None) -> Path:
    if source:
        output_path = Path(base_dir) / source
    else:
        output_path = Path(base_dir)

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path