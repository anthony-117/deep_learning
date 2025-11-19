from typing import Dict, Type, List

from .base import BaseScraper
from .exceptions import SourceNotFoundError, ConfigurationError


class ScraperFactory:
    _registry: Dict[str, Type[BaseScraper]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Args:
            name: Source name (e.g., 'arxiv', 'biorxiv')
        Example:
            @ScraperFactory.register('arxiv')
            class ArxivScraper(BaseScraper):
                pass
        """
        def decorator(scraper_class: Type[BaseScraper]):
            cls._registry[name.lower()] = scraper_class
            return scraper_class
        return decorator

    @classmethod
    def register_class(cls, name: str, scraper_class: Type[BaseScraper]):
        """
        Args:
            name: Source name
            scraper_class: Scraper class to register
        """
        cls._registry[name.lower()] = scraper_class

    @classmethod
    def create(cls, name: str, config) -> BaseScraper:
        """
        Create a scraper instance.

        Args:
            name: Source name (e.g., 'arxiv', 'biorxiv')
            config: ScraperSourceConfig instance

        Returns:
            Scraper instance

        Raises:
            SourceNotFoundError: If scraper not found
            ConfigurationError: If configuration is invalid
        """
        name_lower = name.lower()

        if name_lower not in cls._registry:
            raise SourceNotFoundError(name, cls.list_available())

        try:
            scraper_class = cls._registry[name_lower]
            return scraper_class(config)
        except Exception as e:
            raise ConfigurationError(f"Failed to create scraper '{name}': {str(e)}")

    @classmethod
    def list_available(cls) -> List[str]:
        """
        Get list of available scraper sources.

        Returns:
            List of registered source names
        """
        return sorted(cls._registry.keys())

    @classmethod
    def is_available(cls, name: str) -> bool:
        """
        Check if a scraper is available.

        Args:
            name: Source name

        Returns:
            True if available, False otherwise
        """
        return name.lower() in cls._registry

    @classmethod
    def clear_registry(cls):
        """Clear the scraper registry (mainly for testing)."""
        cls._registry.clear()


# Auto-register all scrapers
def _register_default_scrapers():
    """Register all default scrapers."""
    try:
        from .arxiv_scraper import ArxivScraper
        ScraperFactory.register_class('arxiv', ArxivScraper)
    except ImportError:
        pass

    try:
        from .biorxiv_scraper import BioRxivScraper, MedRxivScraper
        ScraperFactory.register_class('biorxiv', BioRxivScraper)
        ScraperFactory.register_class('medrxiv', MedRxivScraper)
    except ImportError:
        pass

    try:
        from .engrxiv_scraper import EngRxivScraper
        ScraperFactory.register_class('engrxiv', EngRxivScraper)
    except ImportError:
        pass

    try:
        from .chemrxiv_scraper import ChemRxivScraper
        ScraperFactory.register_class('chemrxiv', ChemRxivScraper)
    except ImportError:
        pass


_register_default_scrapers()