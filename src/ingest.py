from typing import Optional, Iterator

from colorama import Fore, Style, init
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from .processing import collect_documents_paths, convert_to_markdown, chunk
from .vectordb import VectorStore

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


def ingest_pipeline(
        directory_path: Optional[str] = None,
        recursive: bool = True,
        file_extensions: Optional[list[str]] = None,
        batch_size: int = 100,
        embedding: Optional[Embeddings] = None,
        drop_old: bool = False,
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

    # Step 3: Chunking
    _print_step("✂️", "Chunking documents with HybridChunker...", Fore.CYAN)
    _print_info("  ℹ️", f"Batch size: {batch_size} chunks")
    chunker = HybridChunker()
    chunk_iter = chunk(iter_document, chunker)

    # Step 4: Initialize embeddings and vector store
    if embedding is None:
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
            vector_store.create_from_documents([doc], drop_old=drop_old)
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


