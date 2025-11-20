"""Node for scraping papers and ingesting into RAG system."""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .graph_state import GraphState
from ..paper_scraping import ScraperManager
from ..ingest_papers import ingest_papers_pipeline2
from ..config import config

def detect_scraping_request(llm: BaseChatModel):
    """
    Creates a detection node that determines whether the user's query requests
    *scraping new research papers* (from external sources) or *searching existing papers*
    (already stored in the local database).

    Args:
        llm (BaseChatModel): Language model used for semantic intent classification.

    Returns:
        Callable[[GraphState], GraphState]: A function that enriches the GraphState with:
            - "needs_scraping" (bool): True if user intends to scrape new papers.
            - "steps" (list[str]): Explanation of the detection decision.
    """

    def _detect(state: GraphState) -> GraphState:
        """
        Detects whether the user is asking to scrape new papers or to search existing ones.
        Combines keyword heuristics with LLM-based classification for improved accuracy.
        """
        question = state.get("question", "").strip()
        question_lower = question.lower()

        # Keywords strongly suggesting scraping behavior
        strong_scraping_keywords = [
            "download papers", "scrape papers", "fetch papers", "get papers",
            "find papers", "collect papers", "retrieve papers", "gather papers",
            "from arxiv", "from biorxiv", "from medrxiv", "from chemrxiv",
            "latest papers", "recent papers", "new papers"
        ]

        # Fast keyword-based filtering before invoking LLM
        has_strong_keyword = any(keyword in question_lower for keyword in strong_scraping_keywords)

        if has_strong_keyword:
            # Structured, minimal, and LLM-friendly classification prompt
            detection_prompt = ChatPromptTemplate.from_template(
                """You are an intent classifier for research queries.

Your task: Determine if the user wants to **SCRAPE NEW PAPERS** (from external repositories)
or **SEARCH EXISTING PAPERS** (in a local/vector database).

User Query:
{question}

Guidelines:
- SCRAPE if the user asks to download, fetch, or retrieve new or latest papers from sources like arxiv, biorxiv, medrxiv, or chemrxiv.
- SEARCH if the user wants to query, explore, or analyze papers already present in the system.

Respond with only one word:
SCRAPE or SEARCH
"""
            )

            chain = detection_prompt | llm | StrOutputParser()
            raw_result = chain.invoke({"question": question}).strip().upper()
            needs_scraping = raw_result == "SCRAPE"
        else:
            needs_scraping = False

        state = {
            **state,
            "needs_scraping": needs_scraping,
            "steps": [
                f"User intent detected as: {'SCRAPE (fetch new papers)' if needs_scraping else 'SEARCH (existing papers)'}"
            ]
        }

        return state

    return _detect


def extract_scraping_params(llm: BaseChatModel):
    """
    Factory function to create a node that extracts scraping parameters from query.

    Args:
        llm: Language model for parameter extraction

    Returns:
        Function that extracts scraping parameters
    """
    def _extract(state: GraphState) -> GraphState:
        """
        Extract scraping parameters from the user's question using LLM.
        """
        question = state["question"]

        extraction_prompt = ChatPromptTemplate.from_template(
            """You are an assistant that extracts paper search parameters from user queries.

User Query: {question}

Extract the following information and return ONLY a JSON object with these fields:
- query: the main search topic/keywords (required)
- sources: list of sources to search from. Choose one or MORE from: ["arxiv", "biorxiv", "medrxiv", "chemrxiv", "engrxiv"]
  * arxiv: Computer science, physics, mathematics, statistics, etc.
  * biorxiv: Biology preprints
  * medrxiv: Medical and health sciences preprints
  * chemrxiv: Chemistry preprints
  * Can select multiple sources - papers will be fetched from EACH source
  * Default: ["arxiv"] if not specified or unclear
- max_results: number of papers to find PER SOURCE (default: 10)
- categories: list of arXiv categories if mentioned (e.g., ["cs.AI", "cs.LG"])
- date_from: start date in YYYY-MM-DD format if mentioned
- date_to: end date in YYYY-MM-DD format if mentioned

Examples:
Query: "Find 20 papers on transformers from arXiv"
Output: {{"query": "transformers", "sources": ["arxiv"], "max_results": 20}}

Query: "Search for recent COVID-19 papers on biorxiv and medrxiv"
Output: {{"query": "COVID-19", "sources": ["biorxiv", "medrxiv"], "max_results": 10}}

Query: "Get machine learning papers from computer science categories"
Output: {{"query": "machine learning", "sources": ["arxiv"], "categories": ["cs.LG", "cs.AI"], "max_results": 10}}

Query: "Find papers on CRISPR gene editing from biology and medical sources"
Output: {{"query": "CRISPR gene editing", "sources": ["biorxiv", "medrxiv"], "max_results": 10}}

Query: "Search for deep learning papers across all computer science and engineering sources"
Output: {{"query": "deep learning", "sources": ["arxiv", "engrxiv"], "max_results": 10}}

Query: "Get papers on drug discovery"
Output: {{"query": "drug discovery", "sources": ["biorxiv", "medrxiv", "chemrxiv"], "max_results": 10}}

Now extract from the user query above. Return ONLY the JSON object, nothing else."""
        )

        chain = extraction_prompt | llm | StrOutputParser()
        result = chain.invoke({"question": question})

        # Parse JSON result
        import json
        try:
            # Clean up the result (remove markdown code blocks if present)
            result = result.strip()
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
            result = result.strip()

            params = json.loads(result)
        except Exception as e:
            # Fallback to basic extraction
            params = {
                "query": question,
                "sources": ["arxiv"],
                "max_results": 10
            }

        return {
            **state,
            "scraping_params": params,
            "steps": [f"Extracted scraping params: {params}"]
        }

    return _extract


def scrape_and_ingest_papers(vector_store_instance):
    """
    Factory function to create a node that scrapes papers and ingests them.

    Args:
        vector_store_instance: VectorStore instance to add papers to

    Returns:
        Function that performs scraping and ingestion
    """
    def _scrape(state: GraphState) -> GraphState:
        """
        Scrape papers based on extracted parameters and ingest into vector store.
        """
        params = state.get("scraping_params", {})

        query = params.get("query", state["question"])
        sources = params.get("sources", ["arxiv"])
        max_results = params.get("max_results", 10)

        # Initialize scraper
        manager = ScraperManager(config.SCRAPER)

        # Build filters
        filters = {}
        if "categories" in params:
            filters["categories"] = params["categories"]
        if "date_from" in params:
            filters["date_from"] = params["date_from"]
        if "date_to" in params:
            filters["date_to"] = params["date_to"]

        # Search for papers
        all_papers = manager.search_all(
            query=query,
            sources=sources,
            max_results=max_results,
            **filters
        )

        # Flatten and deduplicate
        papers_list = []
        for source_papers in all_papers.values():
            papers_list.extend(source_papers)

        unique_papers = manager.deduplicate_results(papers_list, method="doi")

        # Download papers
        download_dir = config.SCRAPER.DOWNLOAD_DIR
        paths = manager.download_papers(
            papers=unique_papers,
            output_dir=download_dir,
            parallel=True
        )

        # Save metadata JSON files alongside each PDF
        import json
        from pathlib import Path

        for path, paper in zip(paths, unique_papers):
            try:
                # Create metadata filename matching the PDF filename
                metadata_file = path.with_name(f"{path.stem}_metadata.json")

                # Save metadata as JSON
                with open(metadata_file, 'w') as f:
                    json.dump(paper.to_dict(), f, indent=2, default=str)

            except Exception as e:
                print(f"Warning: Failed to save metadata for {paper.id}: {e}")

        # Ingest into vector store
        if paths:
            ingest_papers_pipeline2(paths)

        steps = [
            f"Scraped {len(papers_list)} papers from {len(all_papers)} sources",
            f"Found {len(unique_papers)} unique papers",
            f"Downloaded {len(paths)} papers",
            f"Ingested papers into vector database"
        ]

        return {
            **state,
            "scraped_papers": unique_papers,
            "scraped_count": len(unique_papers),
            "steps": steps
        }

    return _scrape


def generate_scraping_summary(llm: BaseChatModel):
    """
    Factory function to create a node that generates a summary after scraping.

    Args:
        llm: Language model for generation

    Returns:
        Function that generates scraping summary
    """
    def _summarize(state: GraphState) -> GraphState:
        """
        Generate a summary of the scraped papers for the user.
        """
        scraped_papers = state.get("scraped_papers", [])
        scraped_count = state.get("scraped_count", 0)
        params = state.get("scraping_params", {})

        # Create summary of papers
        papers_info = []
        for i, paper in enumerate(scraped_papers[:5], 1):  # Show top 5
            authors_str = ", ".join([a.name for a in paper.authors[:2]])
            if len(paper.authors) > 2:
                authors_str += " et al."
            papers_info.append(f"{i}. {paper.title}\n   Authors: {authors_str}\n   Source: {paper.source}")

        papers_summary = "\n\n".join(papers_info)
        if scraped_count > 5:
            papers_summary += f"\n\n... and {scraped_count - 5} more papers"

        summary_prompt = ChatPromptTemplate.from_template(
            """You successfully scraped and ingested papers into the knowledge base.

Search Query: {query}
Sources: {sources}
Papers Found: {count}

Top Papers:
{papers}

Generate a friendly summary for the user explaining:
1. What papers were found and added to the knowledge base
2. That they can now ask questions about these papers
3. A brief mention of the topics covered

Keep it concise and helpful."""
        )

        chain = summary_prompt | llm | StrOutputParser()

        summary = chain.invoke({
            "query": params.get("query", "papers"),
            "sources": ", ".join(params.get("sources", ["arxiv"])),
            "count": scraped_count,
            "papers": papers_summary
        })

        return {
            **state,
            "generation": summary,
            "steps": ["Generated scraping summary for user"]
        }

    return _summarize