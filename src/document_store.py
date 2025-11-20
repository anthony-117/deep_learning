import sqlite3
import json
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from src.paper_scraping import PaperMetadata


class DocumentStore:
    """SQLite-based document metadata store."""

    def __init__(self, db_path: str = "documents.db"):
        """
        Initialize document store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create documents table if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    url TEXT NOT NULL,
                    category TEXT NOT NULL,
                    file_path TEXT NOT NULL UNIQUE,
                    file_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    -- Paper-specific fields (NULL for non-paper documents)
                    source TEXT,
                    paper_id TEXT,
                    title TEXT,
                    authors TEXT,
                    abstract TEXT,
                    published_date TIMESTAMP,
                    updated_date TIMESTAMP,
                    doi TEXT,
                    categories TEXT,
                    keywords TEXT,
                    subjects TEXT,
                    citation_count INTEGER,
                    download_count INTEGER,
                    extra_metadata TEXT
                )
            """)
            # Create index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_path ON documents(file_path)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_paper_id ON documents(paper_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doi ON documents(doi)
            """)

    def insert_document(
        self,
        domain: str,
        url: str,
        category: str,
        file_path: str,
        file_name: str
    ) -> Optional[int]:
        """
        Insert a document into the database.

        Args:
            domain: Domain name (e.g., 'example.com')
            url: Full reconstructed URL
            category: Category (image/text/document)
            file_path: Absolute file path
            file_name: Original filename

        Returns:
            document_id if inserted, None if already exists (skipped)
        """
        with self._get_connection() as conn:
            # Check if document already exists
            cursor = conn.execute(
                "SELECT id FROM documents WHERE file_path = ?",
                (file_path,)
            )
            existing = cursor.fetchone()

            if existing:
                # Document already exists, skip insertion
                return None

            # Insert new document
            cursor = conn.execute("""
                INSERT INTO documents (domain, url, category, file_path, file_name)
                VALUES (?, ?, ?, ?, ?)
            """, (domain, url, category, file_path, file_name))

            return cursor.lastrowid

    def get_document(self, document_id: int) -> Optional[dict]:
        """
        Retrieve document metadata by ID.

        Args:
            document_id: Document ID

        Returns:
            Dictionary with document metadata or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE id = ?",
                (document_id,)
            )
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    def get_document_by_path(self, file_path: str) -> Optional[dict]:
        """
        Retrieve document metadata by file path.

        Args:
            file_path: Absolute file path

        Returns:
            Dictionary with document metadata or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    def insert_paper(
        self,
        paper_metadata: PaperMetadata,
        file_path: str,
        file_name: str
    ) -> Optional[int]:
        """
        Insert a paper with full metadata into the database.

        Args:
            paper_metadata: PaperMetadata object from scrapers
            file_path: Absolute file path to the downloaded PDF
            file_name: Original filename

        Returns:
            document_id if inserted, None if already exists (skipped)
        """
        with self._get_connection() as conn:
            # Check if document already exists by file_path
            cursor = conn.execute(
                "SELECT id FROM documents WHERE file_path = ?",
                (file_path,)
            )
            existing = cursor.fetchone()

            if existing:
                return None

            # Serialize lists and complex objects
            authors_json = json.dumps([
                {
                    'name': a.name,
                    'affiliation': a.affiliation,
                    'orcid': a.orcid
                }
                for a in paper_metadata.authors
            ])
            categories_json = json.dumps(paper_metadata.categories)
            keywords_json = json.dumps(paper_metadata.keywords)
            subjects_json = json.dumps(paper_metadata.subjects)
            extra_metadata_json = json.dumps(paper_metadata.extra_metadata)

            # Insert new paper
            cursor = conn.execute("""
                INSERT INTO documents (
                    domain, url, category, file_path, file_name,
                    source, paper_id, title, authors, abstract,
                    published_date, updated_date, doi,
                    categories, keywords, subjects,
                    citation_count, download_count, extra_metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paper_metadata.source,  # domain (use source as domain)
                paper_metadata.html_url or '',  # url
                'paper',  # category
                file_path,
                file_name,
                paper_metadata.source,
                paper_metadata.id,
                paper_metadata.title,
                authors_json,
                paper_metadata.abstract,
                paper_metadata.published_date.isoformat() if paper_metadata.published_date else None,
                paper_metadata.updated_date.isoformat() if paper_metadata.updated_date else None,
                paper_metadata.doi,
                categories_json,
                keywords_json,
                subjects_json,
                paper_metadata.citation_count,
                paper_metadata.download_count,
                extra_metadata_json
            ))

            return cursor.lastrowid


def parse_file_metadata(file_path: Path) -> dict:
    """
    Parse metadata from file path.

    Expected structure: base_dir/{domain}/{category}/filename
    Example: /data/docs/example.com/document/page_about.pdf

    Args:
        file_path: Path object

    Returns:
        Dictionary with domain, url, category, file_path, file_name
    """
    parts = file_path.parts
    file_name = file_path.name

    # Extract category (parent directory: image/text/document)
    category = file_path.parent.name

    # Extract domain (grandparent directory)
    domain = file_path.parent.parent.name

    # Reconstruct URL from filename
    # Remove file extension and replace underscores with slashes
    url_path = file_path.stem.replace("_", "/")
    url = f"{domain}/{url_path}"

    return {
        "domain": domain,
        "url": url,
        "category": category,
        "file_path": str(file_path.absolute()),
        "file_name": file_name
    }