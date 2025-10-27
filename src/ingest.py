from typing import Optional, Iterator
import json
from pathlib import Path

from colorama import Fore, Style, init
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from langchain_huggingface import HuggingFaceEmbeddings

from .processing import collect_documents_paths, convert_to_markdown, chunk, DocumentMetadata
from .vectordb import VectorStore
from .document_store import DocumentStore

# Initialize colorama
init(autoreset=True)


def _print_step(emoji: str, message: str, color: str = Fore.CYAN):
    """Print a colored step message."""
    print(f"{color}{emoji} {message}{Style.RESET_ALL}")


def _print_success(emoji: str, message: str):
    """Print a success message."""
    print(f"{Fore.GREEN}{emoji} {message}{Style.RESET_ALL}")


def _print_info(emoji: str, message: str):
    """Print an info message."""
    print(f"{Fore.YELLOW}{emoji} {message}{Style.RESET_ALL}")


def _process_and_store_documents(document_iter, doc_store: DocumentStore, storage_dir: str = "documents"):
    """
    Process documents: insert to DB, copy to filesystem storage, update metadata with document_id.

    Args:
        document_iter: Iterator of (DoclingDocument, DocumentMetadata) tuples
        doc_store: DocumentStore instance for DB operations
        storage_dir: Directory to store document files

    Yields:
        Tuple of (DoclingDocument, DocumentMetadata) with updated document_id
    """
    import shutil

    storage_path = Path(storage_dir)
    storage_path.mkdir(exist_ok=True)

    for doc, metadata in document_iter:
        # Insert document to DB and get document_id
        document_id = doc_store.insert_document(
            domain=metadata.domain,
            url=metadata.url,
            category=metadata.file_path.parent.name,  # Extract category from path
            file_path=str(metadata.file_path),
            file_name=metadata.file_name
        )

        if document_id is None:
            # Document already exists, get existing ID
            existing_doc = doc_store.get_document_by_path(str(metadata.file_path))
            document_id = existing_doc["id"] if existing_doc else None

        # Update metadata with document_id
        metadata.document_id = document_id

        # Copy original file to storage with document_id naming
        if document_id:
            # Preserve original file extension
            file_extension = metadata.file_path.suffix
            new_file_path = storage_path / f"doc_{document_id}{file_extension}"

            # Copy the file (use copy2 to preserve metadata)
            shutil.copy2(metadata.file_path, new_file_path)

        yield doc, metadata


def ingest_pipeline(
        directory_path: Optional[str] = None,
        recursive: bool = True,
        file_extensions: Optional[list[str]] = None,
        batch_size: int = 100
) -> VectorStore:
    """
    Complete ingestion pipeline: collect, convert, chunk, embed, and store documents.

    Args:
        directory_path: Path to directory containing documents
        recursive: Whether to search subdirectories
        file_extensions: List of file extensions to process
        batch_size: Number of chunks to collect before adding to vector store

    Returns:
        VectorStore instance with embedded documents
    """
    print(f"\n{Fore.MAGENTA}{'='*70}{Style.RESET_ALL}")
    _print_step("🚀", "Starting Document Ingestion Pipeline", Fore.MAGENTA)
    print(f"{Fore.MAGENTA}{'='*70}{Style.RESET_ALL}\n")

    # Step 1: Collect document paths
    _print_step("📁", f"Collecting documents from: {directory_path}", Fore.CYAN)
    _print_info("  ℹ️", f"Recursive search: {recursive}")
    if file_extensions:
        _print_info("  ℹ️", f"File extensions: {', '.join(file_extensions)}")

    paths = collect_documents_paths(directory_path, recursive, file_extensions)
    _print_success("  ✅", f"Found {len(paths)} document(s)")

    # Step 2: Convert documents to markdown
    _print_step("📄", "Converting documents to markdown format...", Fore.CYAN)
    iter_document = convert_to_markdown(paths)

    # Step 2.5: Initialize document store and store documents
    _print_step("💾", "Initializing document store and storing documents...", Fore.CYAN)
    doc_store = DocumentStore()
    iter_document_with_ids = _process_and_store_documents(iter_document, doc_store)
    _print_success("  ✅", "Document store initialized")

    # Step 3: Chunking
    _print_step("✂️", "Chunking documents with HybridChunker...", Fore.CYAN)
    _print_info("  ℹ️", f"Batch size: {batch_size} chunks")
    chunker = HybridChunker()
    chunk_iter = chunk(iter_document_with_ids, chunker)

    # Step 4: Initialize embeddings and vector store
    _print_step("🧠", "Initializing embedding model...", Fore.CYAN)
    embedding = HuggingFaceEmbeddings()
    _print_success("  ✅", "Embedding model loaded")

    _print_step("🗄️", "Creating vector store...", Fore.CYAN)
    vector_store = VectorStore(embedding)
    _print_success("  ✅", "Vector store initialized")

    # Step 5: Add chunks to vector store in batches
    _print_step("💾", "Adding chunks to vector store in batches...", Fore.CYAN)

    batch = []
    total_chunks = 0
    batch_count = 0

    for doc in chunk_iter:
        # todo don't like this much
        if total_chunks == 0:
            # Create vector store with first document
            vector_store.create_from_documents([doc])
            total_chunks += 1
            continue  # Skip adding to batch, move to next doc

        batch.append(doc)
        total_chunks += 1

        if len(batch) >= batch_size:
            batch_count += 1
            _print_info("  📦", f"Processing batch {batch_count} ({len(batch)} chunks)...")
            vector_store.add_documents(batch)
            _print_success("  ✅", f"Batch {batch_count} added successfully")
            batch = []

    # Add any remaining chunks
    if batch:
        batch_count += 1
        _print_info("  📦", f"Processing final batch ({len(batch)} chunks)...")
        vector_store.add_documents(batch)
        _print_success("  ✅", f"Final batch added successfully")

    # Summary
    print(f"\n{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
    _print_success("🎉", "Ingestion Pipeline Completed Successfully!")
    print(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
    _print_info("📊", f"Total documents processed: {len(paths)}")
    _print_info("📊", f"Total chunks created: {total_chunks}")
    _print_info("📊", f"Total batches processed: {batch_count}")
    print()

    return vector_store


