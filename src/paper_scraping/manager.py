from typing import List, Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import PaperMetadata, BaseScraper
from .factory import ScraperFactory
from .exceptions import SourceNotFoundError
from .utils import deduplicate_papers


class ScraperManager:
    # todo config should be strongly typed
    def __init__(self, config):
        """
        Args:
            config: ScraperConfig instance from main Config
        """
        self.config = config
        self.scrapers: Dict[str, BaseScraper] = {}
        self._initialize_scrapers()

    def _initialize_scrapers(self):
        # Map of source names to config attributes
        source_configs = {
            'arxiv': self.config.ARXIV,
            'biorxiv': self.config.BIORXIV,
            'medrxiv': self.config.MEDRXIV,
            'engrxiv': self.config.ENGRXIV,
            'chemrxiv': self.config.CHEMRXIV,
        }

        for source_name, source_config in source_configs.items():
            if source_config.enabled:
                try:
                    scraper = ScraperFactory.create(source_name, source_config)
                    self.scrapers[source_name] = scraper
                except Exception as e:
                    print(f"Warning: Failed to initialize {source_name} scraper: {e}")

    def search_source(
        self,
        source: str,
        query: str,
        max_results: Optional[int] = None,
        **filters
    ) -> List[PaperMetadata]:
        """
        Args:
            source: Source name (e.g., 'arxiv', 'biorxiv')
            query: Search query string
            max_results: Maximum number of results
            **filters: Source-specific filters

        Returns:
            List of PaperMetadata objects

        Raises:
            SourceNotFoundError: If source not available or not enabled
        """
        source_lower = source.lower()

        if source_lower not in self.scrapers:
            available = list(self.scrapers.keys())
            raise SourceNotFoundError(source, available)

        scraper = self.scrapers[source_lower]
        return scraper.search(query, max_results, **filters)

    def search_all(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        **filters
    ) -> Dict[str, List[PaperMetadata]]:
        """
        Search multiple sources for papers.

        Args:
            query: Search query string
            sources: List of source names (if None, search all enabled sources)
            max_results: Maximum number of results per source
            **filters: Filters to apply to all sources

        Returns:
            Dictionary mapping source names to lists of papers
        """
        # Determine which sources to search
        if sources is None:
            sources_to_search = list(self.scrapers.keys())
        else:
            sources_to_search = [s.lower() for s in sources if s.lower() in self.scrapers]

        results = {}

        if self.config.PARALLEL_SOURCES:
            # Parallel search
            results = self._search_parallel(sources_to_search, query, max_results, **filters)
        else:
            # Sequential search
            for source in sources_to_search:
                try:
                    papers = self.search_source(source, query, max_results, **filters)
                    results[source] = papers
                except Exception as e:
                    print(f"Warning: Search failed for {source}: {e}")
                    results[source] = []

        return results

    def _search_parallel(
        self,
        sources: List[str],
        query: str,
        max_results: Optional[int],
        **filters
    ) -> Dict[str, List[PaperMetadata]]:
        """
        Search multiple sources in parallel.

        Args:
            sources: List of source names
            query: Search query
            max_results: Max results per source
            **filters: Additional filters

        Returns:
            Dictionary of results per source
        """
        results = {}

        with ThreadPoolExecutor(max_workers=len(sources)) as executor:
            # Submit all search tasks
            future_to_source = {
                executor.submit(
                    self.search_source,
                    source,
                    query,
                    max_results,
                    **filters
                ): source
                for source in sources
            }

            # Collect results as they complete
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    papers = future.result()
                    results[source] = papers
                except Exception as e:
                    print(f"Warning: Search failed for {source}: {e}")
                    results[source] = []

        return results

    def download_papers(
        self,
        papers: List[PaperMetadata],
        output_dir: Optional[str] = None,
        parallel: bool = True
    ) -> List[Path]:
        """
        Download PDFs for a list of papers.

        Args:
            papers: List of PaperMetadata objects
            output_dir: Output directory (default: from config)
            parallel: Whether to download in parallel

        Returns:
            List of paths to downloaded PDFs
        """
        if output_dir is None:
            output_dir = self.config.DOWNLOAD_DIR

        if parallel:
            return self._download_parallel(papers, output_dir)
        else:
            return self._download_sequential(papers, output_dir)

    def _download_sequential(
        self,
        papers: List[PaperMetadata],
        output_dir: str
    ) -> List[Path]:
        """Download papers sequentially."""
        paths = []

        for paper in papers:
            try:
                scraper = self.scrapers.get(paper.source)
                if scraper:
                    path = scraper.download_paper(paper, output_dir)
                    paths.append(path)
                else:
                    print(f"Warning: Scraper for {paper.source} not available")
            except Exception as e:
                print(f"Warning: Failed to download {paper.id}: {e}")

        return paths

    def _download_parallel(
        self,
        papers: List[PaperMetadata],
        output_dir: str
    ) -> List[Path]:
        """Download papers in parallel."""
        paths = []
        max_workers = min(5, len(papers))  # Limit concurrent downloads

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_paper = {}

            for paper in papers:
                scraper = self.scrapers.get(paper.source)
                if scraper:
                    future = executor.submit(scraper.download_paper, paper, output_dir)
                    future_to_paper[future] = paper

            for future in as_completed(future_to_paper):
                paper = future_to_paper[future]
                try:
                    path = future.result()
                    paths.append(path)
                except Exception as e:
                    print(f"Warning: Failed to download {paper.id}: {e}")

        return paths

    def deduplicate_results(
        self,
        papers: List[PaperMetadata],
        method: str = "doi"
    ) -> List[PaperMetadata]:
        """
        Deduplicate papers from multiple sources.

        Args:
            papers: List of papers to deduplicate
            method: Deduplication method ('doi', 'title', 'strict')

        Returns:
            Deduplicated list of papers
        """
        if not self.config.DEDUPLICATE_RESULTS:
            return papers

        return deduplicate_papers(papers, method)

    def get_available_sources(self) -> List[str]:
        """
        Get list of available (initialized) sources.

        Returns:
            List of source names
        """
        return list(self.scrapers.keys())

    def get_scraper(self, source: str) -> Optional[BaseScraper]:
        """
        Get a specific scraper instance.

        Args:
            source: Source name

        Returns:
            Scraper instance or None if not found
        """
        return self.scrapers.get(source.lower())

    def __repr__(self) -> str:
        sources = ', '.join(self.scrapers.keys())
        return f"ScraperManager(sources=[{sources}])"