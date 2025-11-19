import json
from pathlib import Path
from typing import List, Optional, Iterator, Tuple

from colorama import Fore, Style
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import PageChunker, BaseChunker
from docling_core.types import DoclingDocument
from langchain_core.documents import Document
from langchain_docling.loader import BaseMetaExtractor, MetaExtractor
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import  config
from src.document_store import DocumentStore
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from src.ingest import _print_info, _print_success
from src.processing import _get_pdf_pipline
from src.vectordb import VectorStore
from src.paper_scraping.base import PaperMetadata

def collect_papers_from_directory(
        papers_dir: Path
) -> Iterator[Tuple[Path, PaperMetadata]]:
    """
    Expected structure:
    papers_dir/
        {source}/
            {paper_id}_{title}.pdf
            {paper_id}_metadata.json
    """
    for source_dir in papers_dir.iterdir():
        if not source_dir.is_dir():
            continue

        for pdf_file in source_dir.glob("*.pdf"):
            # Look for corresponding metadata file
            metadata_file = pdf_file.with_suffix('').parent / f"{pdf_file.stem}_metadata.json"

            if not metadata_file.exists():
                print(f"Warning: No metadata file found for {pdf_file.name}, skipping")
                continue

            # Load metadata
            try:
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                paper_metadata = PaperMetadata(**metadata_dict)
                yield pdf_file, paper_metadata
            except Exception as e:
                print(f"Warning: Failed to load metadata for {pdf_file.name}: {e}")
                continue

def convert_papers_to_markdown(
        paper_iter: Iterator[Tuple[Path, PaperMetadata]],
) -> Iterator[Tuple[DoclingDocument, PaperMetadata]]:

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=_get_pdf_pipline())
        }
    )

    for pdf_path, paper_metadata in paper_iter:
        try:
            # Convert PDF using Docling
            result = doc_converter.convert(str(pdf_path))
            docling_doc = result.document

            yield docling_doc, paper_metadata

        except Exception as e:
            print(f"Warning: Failed to convert {pdf_path.name}: {e}")
            continue

def store_paper_metadata_and_files(
        doc_iter: Iterator[Tuple[DoclingDocument, PaperMetadata]],
        doc_store: DocumentStore,
        storage_dir: Path
) -> Iterator[Tuple[DoclingDocument, PaperMetadata]]:

    storage_dir.mkdir(parents=True, exist_ok=True)

    for doc, metadata in doc_iter:
        try:
            # Generate new file path with document_id (will be set after insert)
            # Temporarily use original file path for insert
            temp_file_path = str(metadata.pdf_url)

            # Insert paper metadata into database
            document_id = doc_store.insert_paper(
                paper_metadata=metadata,
                file_path=temp_file_path,
                file_name=f"{metadata.id}.pdf"
            )

            if document_id is None:
                print(f"Paper {metadata.id} already exists in database, skipping")
                continue

            metadata.document_id = str(document_id)
            # Save metadata JSON alongside
            metadata_file = storage_dir / f"doc_{document_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2, default=str)

            print(f"Stored paper: {metadata.title[:50]}... (doc_id: {document_id})")

            yield doc, metadata

        except Exception as e:
            import traceback
            print(f"Warning: Failed to store paper {metadata.id}: {e}")
            traceback.print_exc()
            continue


def chunk_paper(document_list: Iterator[tuple[DoclingDocument, PaperMetadata]],
          chunker: Optional[BaseChunker] = PageChunker(),
          meta_extractor: Optional[BaseMetaExtractor] = MetaExtractor()) \
        -> Iterator[Document]:

    for doc, metadata in document_list:
        chunk_iter = chunker.chunk(dl_doc=doc)
        for _chunk in chunk_iter:
            base_metadata = meta_extractor.extract_chunk_meta(
                file_path=str(doc.origin.filename),
                chunk=_chunk,
            )
            chunk_doc = Document(
                page_content=chunker.contextualize(chunk=_chunk),
                metadata={
                    'id': metadata.id,
                    'document_id': metadata.document_id,
                    'source': metadata.source,
                    'title': metadata.title,
                    'authors': json.dumps([a.name for a in metadata.authors]),
                    'doi': str(metadata.doi) if metadata.doi else '',
                    'pdf_url': metadata.pdf_url if metadata.pdf_url else '',
                    'categories': json.dumps(metadata.categories),
                    'keywords': json.dumps(metadata.keywords),
                    'published_date': metadata.published_date.isoformat() if metadata.published_date else '',
                    **base_metadata,
                }
            )

            yield chunk_doc

def ingest_papers_pipeline(
        papers_dir: str,
) -> None:
    batch_size = 100
    papers_path = Path(papers_dir)
    storage_path = Path(config.STORAGE_DIR) / "papers"

    # Initialize components
    doc_store = DocumentStore(config.DOCUMENT_STORE_PATH)

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL_ID)

    vector_store = VectorStore(
        embedding=embeddings,
        collection_name="arxiv"
    )
    chunker = HybridChunker(
        tokenizer=config.EMBED_MODEL_ID,
        max_tokens=config.MAX_CHUNK_SIZE
    )

    # Build pipeline
    print(f"Collecting papers from {papers_path}")
    paper_iter = collect_papers_from_directory(papers_path)

    print("Converting papers to Markdown...")
    doc_iter = convert_papers_to_markdown(paper_iter)

    print("Storing metadata and files...")
    doc_iter = store_paper_metadata_and_files(doc_iter, doc_store, storage_path)
    chunk_iter = chunk_paper(doc_iter, chunker)

    print("Chunking and embedding papers...")

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
    _print_info("📊", f"Total chunks created: {total_chunks}")
    _print_info("📊", f"Total batches processed: {batch_count}")
    print()

def ingest_papers_pipeline2(
        paper_paths: List[Path],
) -> None:
    batch_size = 100
    storage_path = Path(config.STORAGE_DIR) / "papers"

    # Initialize components
    doc_store = DocumentStore(config.DOCUMENT_STORE_PATH)

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBED_MODEL_ID)

    vector_store = VectorStore(
        embedding=embeddings,
        collection_name=config.VECTOR_DB_COLLECTION
    )
    chunker = HybridChunker(
        tokenizer=config.EMBED_MODEL_ID,
        max_tokens=config.MAX_CHUNK_SIZE
    )

    # Build pipeline - create iterator from list of paths
    print(f"Collecting papers from {len(paper_paths)} files")

    def paper_path_iterator():
        """Yield (path, metadata) tuples from list of paper paths."""
        for pdf_path in paper_paths:
            # Look for corresponding metadata file
            metadata_file = pdf_path.with_suffix('').parent / f"{pdf_path.stem}_metadata.json"

            if not metadata_file.exists():
                print(f"Warning: No metadata file found for {pdf_path.name}, skipping")
                continue

            # Load metadata
            with open(metadata_file, 'r') as f:
                import json
                metadata_dict = json.load(f)
                metadata = PaperMetadata(**metadata_dict)

            yield (pdf_path, metadata)

    paper_iter = paper_path_iterator()

    print("Converting papers to Markdown...")
    doc_iter = convert_papers_to_markdown(paper_iter)

    print("Storing metadata and files...")
    doc_iter = store_paper_metadata_and_files(doc_iter, doc_store, storage_path)
    chunk_iter = chunk_paper(doc_iter, chunker)

    print("Chunking and embedding papers...")

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
    _print_info("📊", f"Total chunks created: {total_chunks}")
    _print_info("📊", f"Total batches processed: {batch_count}")
    print()
