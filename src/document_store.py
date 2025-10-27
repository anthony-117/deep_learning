import sqlite3
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Create index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_path ON documents(file_path)
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