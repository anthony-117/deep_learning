__version__ = "1.0.0"
__author__ = "Paper Scraping System"

from .base import BaseScraper, PaperMetadata, Author
from .factory import ScraperFactory
from .manager import ScraperManager
from .exceptions import (
    ScraperException,
    SourceNotFoundError,
    RateLimitError,
    DownloadError,
    InvalidQueryError,
    APIError,
    ConfigurationError
)

# Scraper implementations (optional - can be imported directly)
from .arxiv_scraper import ArxivScraper
from .biorxiv_scraper import BioRxivScraper, MedRxivScraper
from .engrxiv_scraper import EngRxivScraper
from .chemrxiv_scraper import ChemRxivScraper

__all__ = [
    # Core classes
    'ScraperManager',
    'ScraperFactory',
    'BaseScraper',

    # Models
    'PaperMetadata',
    'Author',

    # Exceptions
    'ScraperException',
    'SourceNotFoundError',
    'RateLimitError',
    'DownloadError',
    'InvalidQueryError',
    'APIError',
    'ConfigurationError',

    # Scrapers
    'ArxivScraper',
    'BioRxivScraper',
    'MedRxivScraper',
    'EngRxivScraper',
    'ChemRxivScraper',
]


# Convenience function
def create_manager(config=None):
    if config is None:
        from ..config import config as default_config
        config = default_config.SCRAPER

    return ScraperManager(config)