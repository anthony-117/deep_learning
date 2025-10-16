from pathlib import Path
from typing import Optional, Iterator

from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling_core.transforms.chunker import BaseChunker, PageChunker
from docling_core.types import DoclingDocument
from langchain_docling.loader import BaseMetaExtractor, MetaExtractor
from langchain_core.documents import Document

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions

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


def chunk(document_list: Iterator[DoclingDocument],
          chunker: Optional[BaseChunker] = PageChunker,
          meta_extractor: Optional[BaseMetaExtractor] = MetaExtractor) \
        -> Iterator[Document]:
    for doc in document_list:
        chunk_iter = chunker.chunk(doc)
        for _chunk in chunk_iter:
            yield Document(
                page_content=chunker.contextualize(chunk=_chunk),
                metadata=meta_extractor.extract_chunk_meta(
                    file_path=doc.name,  # todo change this later
                    chunk=_chunk,
                ),
            )

def convert_to_markdown(documents: list[Path]) -> Iterator[DoclingDocument]:
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
                yield result.document

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
