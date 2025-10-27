from pathlib import Path
from typing import Optional, Iterator

from pydantic import BaseModel
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling_core.transforms.chunker import BaseChunker, PageChunker
from docling_core.types import DoclingDocument
from langchain_docling.loader import BaseMetaExtractor, MetaExtractor
from langchain_core.documents import Document

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions


class DocumentMetadata(BaseModel):
    """Metadata extracted from file path and used throughout the pipeline."""
    file_path: Path
    file_name: str
    domain: str
    url: str
    document_id: Optional[int] = None

    @classmethod
    def from_path(cls, file_path: Path) -> "DocumentMetadata":
        """
        Create metadata from file path.

        Expected structure: base_dir/{domain}/{category}/filename
        Example: /data/docs/example.com/document/page_about.pdf

        Args:
            file_path: Path object

        Returns:
            DocumentMetadata instance
        """
        file_name = file_path.name
        domain = file_path.parent.parent.name

        # Reconstruct URL from filename (remove extension, replace _ with /)
        url_path = file_path.stem.replace("_", "/")
        url = f"{domain}/{url_path}"

        return cls(
            file_path=file_path,
            file_name=file_name,
            domain=domain,
            url=url
        )

DEFAULT_EXTENSIONS = ["."+ fmt.value for fmt in InputFormat]

def collect_documents_paths(
        directory_path: Optional[str] = None,
        recursive: bool = True,
        file_extensions: Optional[list[str]] = None
) -> list[Path]:
    if file_extensions is None:
        file_extensions = DEFAULT_EXTENSIONS

    if not directory_path:
        raise ValueError("Either file_path, directory_path, or config.FILE_PATH must be provided")

    dir_path = Path(directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Invalid directory path: {directory_path}")

    return _collect_files(dir_path, file_extensions, recursive)


def chunk(document_list: Iterator[tuple[DoclingDocument, DocumentMetadata]],
          chunker: Optional[BaseChunker] = PageChunker,
          meta_extractor: Optional[BaseMetaExtractor] = MetaExtractor) \
        -> Iterator[Document]:
    for doc, doc_metadata in document_list:
        chunk_iter = chunker.chunk(doc)
        for _chunk in chunk_iter:
            # Extract base chunk metadata from docling
            base_metadata = meta_extractor.extract_chunk_meta(
                file_path=str(doc_metadata.file_path),
                chunk=_chunk,
            )

            # Merge with our custom metadata
            base_metadata.update(doc_metadata.model_dump(exclude={"document_id"}))

            chunk_doc = Document(
                page_content=chunker.contextualize(chunk=_chunk),
                metadata=base_metadata,
            )

            yield chunk_doc

def convert_to_markdown(documents: list[Path]) -> Iterator[tuple[DoclingDocument, DocumentMetadata]]:
    if not documents:
        raise ValueError("Documents list cannot be empty")

    # Create document converter with format-specific options
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=_get_pdf_pipline())
        }
    )

    for doc in documents:
        try:
            result = converter.convert(doc)

            if not result.errors:
                metadata = DocumentMetadata.from_path(doc)
                yield result.document, metadata

        except Exception:
            continue


def _get_pdf_pipline() -> PdfPipelineOptions:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True
    pipeline_options.ocr_options = TesseractCliOcrOptions()

    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.AUTO
    )
    return pipeline_options


def _collect_files(directory: Path, extensions: list[str], recursive: bool) -> list[Path]:
    files = []
    pattern_func = directory.rglob if recursive else directory.glob
    for ext in extensions:
        files.extend(pattern_func(f"*{ext}"))
    return files
