from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from docling.chunking import HybridChunker

from .config import config

def load_document() -> list[Document]:
    loader = DoclingLoader(
        file_path=config.FILE_PATH,
        export_type=config.EXPORT_TYPE,
        chunker=HybridChunker(tokenizer=config.EMBED_MODEL_ID),
    )

    return loader.load()

def chunk(document_list: list[Document]) -> list[Document]:
    #todo i don't like this much
    if config.EXPORT_TYPE == ExportType.DOC_CHUNKS:
        splits = document_list
    elif config.EXPORT_TYPE == ExportType.MARKDOWN:

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ],
            strip_headers=False,
        )
        splits = [split for doc in document_list for split in splitter.split_text(doc.page_content)]
    else:
        raise ValueError(f"Unexpected export type: {config.EXPORT_TYPE}")
    return splits
