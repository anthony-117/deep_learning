from typing import Optional, Iterator

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from .processing import collect_documents_paths, convert_to_markdown, chunk
from .vectordb import VectorStore

def ingest_pipeline(
        directory_path: Optional[str] = None,
        recursive: bool = True,
        file_extensions: Optional[list[str]] = None,
        vector_store: Optional[VectorStore] = None,
        batch_size: int = 100
) -> VectorStore:
    """
    Complete ingestion pipeline: collect, convert, chunk, embed, and store documents.

    Args:
        directory_path: Path to directory containing documents
        recursive: Whether to search subdirectories
        file_extensions: List of file extensions to process
        vector_store: Pre-initialized VectorStore instance (must have vectorstore created)
        batch_size: Number of chunks to collect before adding to vector store

    Returns:
        VectorStore instance with embedded documents
    """
    if not vector_store:
        raise ValueError("vector_store parameter is required and must be pre-initialized")

    paths = collect_documents_paths(directory_path, recursive, file_extensions)
    iter_document = convert_to_markdown(paths)

    # chunking
    chunker = HybridChunker()
    chunk_iter = chunk(iter_document, chunker)

    # Add chunks to vector store in batches as they're generated
    batch = []
    for doc in chunk_iter:
        batch.append(doc)
        if len(batch) >= batch_size:
            vector_store.add_documents(batch)
            batch = []

    # Add any remaining chunks
    if batch:
        vector_store.add_documents(batch)

    return vector_store


