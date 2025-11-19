from typing import List, Optional
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

from .base import BaseScraper, PaperMetadata, Author
from .exceptions import InvalidQueryError, DownloadError, APIError
from .utils import (
    RateLimiter, SimpleCache, sanitize_filename,
    normalize_date, retry_on_failure, create_output_directory
)


class BioRxivBaseScraper(BaseScraper):
    def __init__(self, config, server: str):
        self.server = server
        super().__init__(config)
        # Use server-specific website URL for web scraping
        self.base_url = f"https://www.{server}.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })

        self.rate_limiter = RateLimiter(
            requests_per_period=config.rate_limit_requests,
            period_seconds=config.rate_limit_period
        )

        if config.cache_ttl > 0:
            self.cache = SimpleCache(ttl=config.cache_ttl)
        else:
            self.cache = None

    def _get_name(self) -> str:
        """Get scraper name."""
        return self.server

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        **filters
    ) -> List[PaperMetadata]:
        """
        Search by scraping website search results and fetching full metadata for each paper.

        Args:
            query: Search query string
            max_results: Maximum number of results (default: from config)
            **filters: Additional filters (not used in web scraping)

        Returns:
            List of PaperMetadata objects with complete metadata
        """
        if not self.validate_query(query):
            raise InvalidQueryError(query, "Query cannot be empty")

        # Check cache
        cache_key = f"{query}:{max_results}:{filters}"
        if self.cache:
            cached_results = self.cache.get(cache_key)
            if cached_results:
                return cached_results

        if max_results is None:
            max_results = self.config.max_results

        # Scrape search results and fetch full metadata for each
        papers = self._scrape_search_results(query, max_results)

        # Cache results
        if self.cache:
            self.cache.set(cache_key, papers)

        return papers

    def _scrape_search_results(
        self,
        query: str,
        max_results: int
    ) -> List[PaperMetadata]:
        """
        Scrape search results and fetch full metadata for each paper.

        Args:
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of PaperMetadata objects
        """
        papers = []

        # URL encode the query
        encoded_query = quote(query)
        url = f"{self.base_url}/search/{encoded_query}"

        # Rate limiting
        self.rate_limiter.acquire(self.name)

        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all search result items
            search_results = soup.find_all('li', class_='search-result')

            for result in search_results:
                if len(papers) >= max_results:
                    break

                # Extract the paper link from search result
                paper_link = self._extract_paper_link(result)
                if not paper_link:
                    continue

                # Fetch full metadata from the paper's page
                paper = self.get_paper_metadata(paper_link)
                if paper:
                    papers.append(paper)

        except requests.exceptions.HTTPError as e:
            raise APIError(self.name, status_code=e.response.status_code, message=str(e))
        except Exception as e:
            raise APIError(self.name, message=str(e))

        return papers

    def _extract_paper_link(self, result_elem) -> Optional[str]:
        """
        Extract paper link from search result element.

        Args:
            result_elem: BeautifulSoup element for search result

        Returns:
            Paper URL or None
        """
        try:
            title_elem = result_elem.find('span', class_='highwire-cite-title')
            if not title_elem:
                return None

            link = title_elem.find('a')
            if not link:
                return None

            href = link.get('href', '')
            if href.startswith('/'):
                href = f"{self.base_url}{href}"

            return href

        except Exception:
            return None

    @retry_on_failure(max_attempts=3, delay=1.0, backoff=2.0)
    def download_paper(
        self,
        paper: PaperMetadata,
        output_dir: str
    ) -> Path:
        """
        Download paper PDF.

        Args:
            paper: PaperMetadata object with pdf_url
            output_dir: Directory to save PDF

        Returns:
            Path to downloaded PDF
        """
        if not paper.pdf_url:
            raise DownloadError(paper.id, self.name, "No PDF URL available")

        # Create output directory
        output_path = create_output_directory(output_dir, self.name)

        # Generate filename
        filename = sanitize_filename(f"{paper.id}_{paper.title[:50]}.pdf")
        file_path = output_path / filename

        # Check if already downloaded
        if file_path.exists():
            return file_path

        # Rate limiting
        self.rate_limiter.acquire(self.name)

        try:
            response = self.session.get(
                paper.pdf_url,
                timeout=self.config.timeout,
                stream=True
            )
            response.raise_for_status()

            # Save PDF
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return file_path

        except Exception as e:
            raise DownloadError(paper.id, self.name, str(e))

    def get_paper_metadata(self, paper_url: str) -> Optional[PaperMetadata]:
        """
        Scrape full metadata from a paper's page.

        Args:
            paper_url: Full URL to paper page

        Returns:
            PaperMetadata object or None
        """
        # Rate limiting
        self.rate_limiter.acquire(self.name)

        try:
            response = self.session.get(paper_url, timeout=self.config.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            return self._parse_paper_page(soup, paper_url)

        except requests.exceptions.HTTPError as e:
            print(f"Warning: HTTP error {e.response.status_code} for {paper_url}")
            return None
        except Exception as e:
            print(f"Warning: Failed to fetch paper metadata from {paper_url}: {e}")
            return None

    def _parse_paper_page(self, soup: BeautifulSoup, paper_url: str) -> Optional[PaperMetadata]:
        """
        Parse paper metadata from page HTML.

        Args:
            soup: BeautifulSoup object of paper page
            paper_url: Paper URL

        Returns:
            PaperMetadata object or None
        """
        try:
            # Extract title
            title_elem = soup.find('h1', id='page-title')
            title = title_elem.get_text(strip=True) if title_elem else 'Unknown Title'

            # Extract DOI and paper ID
            doi = None
            paper_id = None
            doi_elem = soup.find('span', class_='highwire-cite-metadata-doi')
            if doi_elem:
                doi_text = doi_elem.get_text(strip=True)
                # Remove both 'doi:' prefix and 'https://doi.org/' prefix
                doi = doi_text.replace('doi:', '').replace('https://doi.org/', '').strip()
                # Extract ID from DOI (e.g., 10.1101/2024.08.20.24312181 -> 2024.08.20.24312181)
                if doi:
                    paper_id = doi.split('/')[-1]

            # Extract authors
            authors = []
            contrib_group = soup.find('div', class_='contributors')
            if contrib_group:
                author_elems = contrib_group.find_all('span', class_='contrib-author')
                for author_elem in author_elems:
                    author_name_elem = author_elem.find('span', class_='name')
                    if author_name_elem:
                        name = author_name_elem.get_text(strip=True)
                        authors.append(Author(name=name))

            # Extract abstract
            abstract = ''
            abstract_elem = soup.find('div', class_='abstract')
            if abstract_elem:
                # Get all paragraph text
                paragraphs = abstract_elem.find_all('p')
                abstract = ' '.join([p.get_text(strip=True) for p in paragraphs])

            # Extract published date
            published_date = None
            date_elem = soup.find('span', class_='highwire-cite-metadata-date')
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                published_date = normalize_date(date_text)

            # Extract categories/subjects
            categories = []
            subject_elems = soup.find_all('span', class_='highwire-article-collection-term')
            for subj in subject_elems:
                categories.append(subj.get_text(strip=True))

            # Construct PDF URL
            pdf_url = None
            if doi:
                pdf_url = f"{self.base_url}/content/{doi}.full.pdf"

            return PaperMetadata(
                id=paper_id or 'unknown',
                source=self.name,
                title=title,
                authors=authors,
                abstract=abstract,
                published_date=published_date,
                updated_date=None,
                pdf_url=pdf_url,
                html_url=paper_url,
                doi=doi,
                categories=categories,
                keywords=[],
                extra_metadata={
                    'scraped_from': 'paper_page'
                }
            )

        except Exception as e:
            print(f"Warning: Failed to parse paper page: {e}")
            return None


class BioRxivScraper(BioRxivBaseScraper):
    """Scraper for bioRxiv preprints."""

    def __init__(self, config):
        super().__init__(config, server='biorxiv')


class MedRxivScraper(BioRxivBaseScraper):
    """Scraper for medRxiv preprints."""

    def __init__(self, config):
        super().__init__(config, server='medrxiv')