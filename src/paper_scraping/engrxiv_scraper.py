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


class EngRxivScraper(BaseScraper):
    def __init__(self, config):
        super().__init__(config)
        self.base_url = "https://api.osf.io/v2"
        self.provider = "engrxiv"
        self.session = requests.Session()

        # Set up headers
        headers = {
            'User-Agent': 'PaperScraper/1.0 (Educational Purpose)',
            'Accept': 'application/vnd.api+json'
        }

        # Add API key if provided
        if config.api_key:
            headers['Authorization'] = f'Bearer {config.api_key}'

        self.session.headers.update(headers)

        self.rate_limiter = RateLimiter(
            requests_per_period=config.rate_limit_requests,
            period_seconds=config.rate_limit_period
        )

        if config.cache_ttl > 0:
            self.cache = SimpleCache(ttl=config.cache_ttl)
        else:
            self.cache = None

    def _get_name(self) -> str:
        return "engrxiv"

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        **filters
    ) -> List[PaperMetadata]:
        """
        Args:
            query: Search query string
            max_results: Maximum number of results (default: from config)
            **filters: Additional filters:
                - date_from: str - Start date (YYYY-MM-DD)
                - date_to: str - End date (YYYY-MM-DD)
                - tags: List[str] - Filter by tags
        """
        if not self.validate_query(query):
            raise InvalidQueryError(query, "Query cannot be empty")

        # Check cache
        cache_key = f"{query}:{max_results}:{filters}"
        if self.cache:
            cached_results = self.cache.get(cache_key)
            if cached_results:
                return cached_results

        # Build search URL
        url = f"{self.base_url}/preprints/"

        # Build query parameters
        params = {
            'filter[provider]': self.provider,
            'filter[is_published]': 'true',
            'filter[reviews_state][ne]': 'initial',  # Exclude initial review state
        }

        # Add text search (searches title and description)
        if query:
            params['filter[title,description]'] = query

        # Add date filters
        date_from = filters.get('date_from')
        date_to = filters.get('date_to')
        if date_from:
            params['filter[date_created][gte]'] = date_from
        if date_to:
            params['filter[date_created][lte]'] = date_to

        # Add pagination
        if max_results is None:
            max_results = self.config.max_results

        params['page[size]'] = min(max_results, 100)  # OSF max page size is 100

        # Rate limiting
        self.rate_limiter.acquire(self.name)

        try:
            response = self.session.get(url, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()

            papers = []
            if 'data' in data:
                for item in data['data']:
                    paper = self._convert_to_paper_metadata(item)
                    papers.append(paper)

                    if len(papers) >= max_results:
                        break

            # Handle pagination if needed
            while 'links' in data and 'next' in data['links'] and len(papers) < max_results:
                next_url = data['links']['next']
                if not next_url:
                    break

                # Rate limiting for pagination
                self.rate_limiter.acquire(self.name)

                response = self.session.get(next_url, timeout=self.config.timeout)
                response.raise_for_status()
                data = response.json()

                if 'data' in data:
                    for item in data['data']:
                        paper = self._convert_to_paper_metadata(item)
                        papers.append(paper)

                        if len(papers) >= max_results:
                            break

            # Cache results
            if self.cache:
                self.cache.set(cache_key, papers)

            return papers

        except requests.exceptions.HTTPError as e:
            raise APIError(self.name, status_code=e.response.status_code, message=str(e))
        except Exception as e:
            raise APIError(self.name, message=str(e))

    @retry_on_failure(max_attempts=3, delay=1.0, backoff=2.0)
    def download_paper(
        self,
        paper: PaperMetadata,
        output_dir: str
    ) -> Path:
        """
        Args:
            paper: PaperMetadata object with pdf_url
            output_dir: Directory to save PDF
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
        Args:
            paper_id: OSF preprint ID

        Returns:
            PaperMetadata object
        """
        url = f"{self.base_url}/preprints/{paper_id}/"

        # Rate limiting
        self.rate_limiter.acquire(self.name)

        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()

            if 'data' in data:
                return self._convert_to_paper_metadata(data['data'])
            else:
                raise APIError(self.name, message=f"Paper {paper_id} not found")

        except requests.exceptions.HTTPError as e:
            raise APIError(self.name, status_code=e.response.status_code, message=str(e))
        except Exception as e:
            raise APIError(self.name, message=str(e))

    def _convert_to_paper_metadata(self, item: Dict[str, Any]) -> PaperMetadata:
        """
        Args:
            item: Dictionary from OSF API response
        """
        attributes = item.get('attributes', {})

        # Extract ID
        paper_id = item.get('id', 'unknown')

        # Extract basic info
        title = attributes.get('title', 'Unknown Title')
        abstract = attributes.get('description', '')

        # Parse dates
        published_date = normalize_date(attributes.get('date_published'))
        updated_date = normalize_date(attributes.get('date_modified'))

        # Extract DOI
        doi = None
        doi_url = attributes.get('doi')
        if doi_url:
            # Extract DOI from URL (format: https://doi.org/10.31224/osf.io/xxxxx)
            doi = doi_url.replace('https://doi.org/', '')

        # Build URLs
        relationships = item.get('relationships', {})
        links = item.get('links', {})

        html_url = links.get('html')
        preprint_doi = links.get('preprint_doi')

        # Get PDF URL from primary file
        # OSF API provides download link in relationships.primary_file.links.download
        # OR in links.download
        pdf_url = None

        # Try from primary_file relationship first
        primary_file = relationships.get('primary_file', {})
        if 'links' in primary_file and 'download' in primary_file['links']:
            pdf_url = primary_file['links']['download']

        # Fallback to direct links
        if not pdf_url and 'download' in links:
            pdf_url = links.get('download')

        # Another fallback: construct from preprint DOI
        if not pdf_url and preprint_doi:
            # OSF format: https://osf.io/preprints/engrxiv/{preprint_id}/download
            pdf_url = f"https://osf.io/preprints/engrxiv/{paper_id}/download"

        # Extract tags/subjects
        tags = attributes.get('tags', [])
        subjects = []
        if 'subjects' in attributes:
            for subject in attributes['subjects']:
                if isinstance(subject, dict) and 'text' in subject:
                    subjects.append(subject['text'])
                elif isinstance(subject, str):
                    subjects.append(subject)

        # Authors - need to fetch from contributors relationship
        # For simplicity, we'll leave this as extra_metadata
        authors = []

        return PaperMetadata(
            id=paper_id,
            source=self.name,
            title=title,
            authors=authors,  # OSF requires additional API call for detailed author info
            abstract=abstract,
            published_date=published_date,
            updated_date=updated_date,
            pdf_url=pdf_url,
            html_url=html_url,
            doi=doi,
            keywords=tags,
            subjects=subjects,
            extra_metadata={
                'license': attributes.get('license', {}).get('name'),
                'is_published': attributes.get('is_published'),
                'reviews_state': attributes.get('reviews_state'),
                'preprint_doi_url': preprint_doi,
                'contributors_url': relationships.get('contributors', {}).get('links', {}).get('related', {}).get('href')
            }
        )