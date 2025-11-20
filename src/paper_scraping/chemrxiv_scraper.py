from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import requests

from .base import BaseScraper, PaperMetadata, Author
from .exceptions import InvalidQueryError, DownloadError, APIError
from .utils import (
    RateLimiter, SimpleCache, sanitize_filename,
    normalize_date, retry_on_failure, create_output_directory
)


class ChemRxivScraper(BaseScraper):
    """Scraper for ChemRxiv preprints using Cambridge Open Engage Public API v1."""

    # ChemRxiv categories - name: id mapping
    CATEGORIES = {
        'Analytical Chemistry': '605c72ef153207001f6470d5',
        'Biological and Medicinal Chemistry': '605c72ef153207001f6470d0',
        'Catalysis': '605c72ef153207001f6470d4',
        'Chemical Education': '605c72ef153207001f6470dc',
        'Chemical Engineering and Industrial Chemistry': '605c72ef153207001f6470db',
        'Earth, Space, and Environmental Chemistry': '605c72ef153207001f6470da',
        'Energy': '605c72ef153207001f6470d9',
        'Inorganic Chemistry': '605c72ef153207001f6470d3',
        'Materials Chemistry': '60b63c9f57d3ab002262a6f7',
        'Materials Science': '605c72ef153207001f6470d2',
        'Nanoscience': '605c72ef153207001f6470d8',
        'Organic Chemistry': '605c72ef153207001f6470d1',
        'Organometallic Chemistry': '605c72ef153207001f6470d6',
        'Physical Chemistry': '605c72ef153207001f6470cf',
        'Polymer Science': '605c72ef153207001f6470d7',
        'Theoretical and Computational Chemistry': '605c72ef153207001f6470ce',
        'Agriculture and Food Chemistry': '605c72ef153207001f6470dd',
    }

    def __init__(self, config):
        super().__init__(config)
        # ChemRxiv uses Cambridge Open Engage Public API v1
        self.base_url = "https://chemrxiv.org/engage/chemrxiv/public-api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PaperScraper/1.0 (Educational Purpose)',
            'Accept': 'application/json'
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
        return "chemrxiv"

    def get_supported_categories(self) -> List[str]:
        """Get list of supported category names."""
        return list(self.CATEGORIES.keys())

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        **filters
    ) -> List[PaperMetadata]:
        """
        Search ChemRxiv for papers.

        Args:
            query: Search query string
            max_results: Maximum number of results (default: from config)
            **filters: Additional filters:
                - date_from: str - Start date (YYYY-MM-DD)
                - date_to: str - End date (YYYY-MM-DD)
                - category: str - Category filter

        Returns:
            List of PaperMetadata objects
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

        # Search using items endpoint
        papers = self._search_items(query, max_results, **filters)

        # Cache results
        if self.cache:
            self.cache.set(cache_key, papers)

        return papers

    def _search_items(
        self,
        query: str,
        max_results: int,
        **filters
    ) -> List[PaperMetadata]:
        """
        Search items using Open Engage API.

        Args:
            query: Search query
            max_results: Maximum results to return
            **filters: Additional filters

        Returns:
            List of PaperMetadata objects
        """
        url = f"{self.base_url}/items"

        # Build query parameters
        params = {
            "term": query,
            "limit": min(max_results, 100),  # Typical API limit per page
            "skip": 0,
            "sort": "RELEVANT_DESC",
        }

        # Add filters - accept both category name and ID
        if 'category' in filters:
            category = filters['category']
            # If category is a name, convert to ID
            if category in self.CATEGORIES:
                params['category'] = self.CATEGORIES[category]
            else:
                # Assume it's already an ID
                params['category'] = category

        if 'date_from' in filters:
            params['publishedDateFrom'] = filters['date_from']

        if 'date_to' in filters:
            params['publishedDateTo'] = filters['date_to']

        papers = []

        while len(papers) < max_results:
            # Rate limiting
            self.rate_limiter.acquire(self.name)

            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                data = response.json()

                # Handle different response formats
                items = data.get('itemHits', []) or data.get('items', []) or data

                if not items:
                    break

                # Convert items to papers
                for item_wrapper in items:
                    if len(papers) >= max_results:
                        break

                    # Extract the actual item from the wrapper
                    item = item_wrapper.get('item', item_wrapper)
                    paper = self._convert_to_paper_metadata(item)
                    papers.append(paper)

                # Check if we should continue pagination
                if len(items) < params['limit']:
                    break

                # Move to next page
                params['skip'] += params['limit']

            except requests.exceptions.HTTPError as e:
                raise APIError(self.name, status_code=e.response.status_code, message=str(e))
            except Exception as e:
                raise APIError(self.name, message=str(e))

        return papers

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

    def get_paper_metadata(self, paper_id: str) -> PaperMetadata:
        """
        Get metadata for a specific paper by ID.

        Args:
            paper_id: Item ID

        Returns:
            PaperMetadata object
        """
        url = f"{self.base_url}/items/{paper_id}"

        # Rate limiting
        self.rate_limiter.acquire(self.name)

        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()

            return self._convert_to_paper_metadata(data)

        except requests.exceptions.HTTPError as e:
            raise APIError(self.name, status_code=e.response.status_code, message=str(e))
        except Exception as e:
            raise APIError(self.name, message=str(e))

    def _convert_to_paper_metadata(self, item: Dict[str, Any]) -> PaperMetadata:
        """
        Convert Open Engage API response to PaperMetadata.

        Args:
            item: Dictionary from API response

        Returns:
            PaperMetadata object
        """
        # Parse authors
        authors = []
        for author_data in item.get('authors', []):
            # Get all institution names and join with comma
            institutions = author_data.get('institutions', [])
            affiliation = ', '.join([inst.get('name', '') for inst in institutions]) if institutions else None

            first_name = author_data.get('firstName', '').strip()
            last_name = author_data.get('lastName', '').strip()
            full_name = f"{first_name} {last_name}".strip()

            author = Author(
                name=full_name or 'Unknown',
                affiliation=affiliation,
                orcid=author_data.get('orcid')
            )
            authors.append(author)

        # Parse dates
        published_date = normalize_date(item.get('publishedDate'))
        updated_date = normalize_date(item.get('statusDate'))

        # Get DOI
        doi = item.get('doi')

        # Get PDF URL from asset (singular)
        pdf_url = None
        asset = item.get('asset')
        if asset and asset.get('mimeType') == 'application/pdf':
            pdf_url = asset.get('original', {}).get('url')

        # Build HTML URL from DOI or item ID
        html_url = None
        if doi:
            html_url = f"https://chemrxiv.org/engage/chemrxiv/article-details/{item.get('id')}"

        # Extract categories
        categories = []
        if 'categories' in item:
            for cat in item['categories']:
                categories.append(cat.get('name', ''))

        # Extract keywords
        keywords = item.get('keywords', [])

        # Extract metrics
        metrics_data = item.get('metrics', [])
        citation_count = None
        download_count = None
        for metric in metrics_data:
            if metric.get('description') == 'Citations':
                citation_count = metric.get('value')
            elif metric.get('description') == 'Content Downloads':
                download_count = metric.get('value')

        return PaperMetadata(
            id=str(item.get('id', 'unknown')),
            source=self.name,
            title=item.get('title', 'Unknown Title'),
            authors=authors,
            abstract=item.get('abstract', ''),
            published_date=published_date,
            updated_date=updated_date,
            pdf_url=pdf_url,
            html_url=html_url,
            doi=doi,
            categories=categories,
            keywords=keywords,
            citation_count=citation_count,
            download_count=download_count,
            extra_metadata={
                'status': item.get('status'),
                'version': item.get('version'),
                'license': item.get('license', {}).get('name'),
                'license_url': item.get('license', {}).get('url'),
                'content_type': item.get('contentType', {}).get('name'),
                'funders': item.get('funders', []),
                'has_competing_interests': item.get('hasCompetingInterests')
            }
        )