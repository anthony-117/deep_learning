# LangGraph RAG Integration Guide

## Overview

This project now includes **LangGraph**, a powerful framework for building stateful, multi-step LLM applications. The LangGraph integration provides an advanced RAG (Retrieval-Augmented Generation) workflow with self-correction, query rewriting, and hallucination detection capabilities.

## What is LangGraph?

LangGraph extends LangChain by enabling the creation of cyclic computational graphs. Unlike linear chains, LangGraph allows:

- **Stateful workflows**: Maintain context across multiple steps
- **Conditional branching**: Make decisions based on intermediate results
- **Cyclic graphs**: Loop back to previous steps for self-correction
- **Multi-agent systems**: Coordinate multiple AI agents

## Architecture

The LangGraph RAG workflow (`src/graph.py`) implements an advanced document Q&A system with the following flow:

```
┌─────────────────┐
│  Analyze Query  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Retrieve     │
│   Documents     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Grade       │
│   Documents     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Relevant   No Relevant
Docs?      Docs?
    │         │
    │         ▼
    │    ┌──────────┐
    │    │ Rewrite  │◄─┐
    │    │  Query   │  │
    │    └────┬─────┘  │
    │         │        │
    │         └────────┘
    │
    ▼
┌─────────────────┐
│    Generate     │
│     Answer      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      Check      │
│  Hallucination  │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
Grounded   Hallucinated
    │         │
    │         ▼
    │    Rewrite (if attempts left)
    │
    ▼
┌─────────────────┐
│      END        │
└─────────────────┘
```

## Key Features

### 1. Query Analysis
Analyzes the user's question to understand intent and optimize for retrieval.

### 2. Document Retrieval
Retrieves relevant documents from the vector store using semantic search.

### 3. Relevance Grading
Uses an LLM to grade each retrieved document for relevance to the question. Filters out irrelevant documents.

### 4. Answer Generation
Generates a concise answer based only on the relevant documents.

### 5. Hallucination Detection
Checks if the generated answer is grounded in the retrieved documents. Prevents the model from making up information.

### 6. Query Rewriting
If documents are irrelevant or the answer is hallucinated, rewrites the query and tries again (up to a configurable limit).

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Key new dependencies:
   - `langgraph~=0.2.62`: Core LangGraph framework
   - `langchain-huggingface~=0.1.2`: HuggingFace integration

2. **Set up environment variables** (in `.env` file):
   ```bash
   # Required for LLM
   HF_TOKEN=your_huggingface_token
   GEN_MODEL_ID=your_model_id

   # Vector database
   VECTOR_URI=your_vector_db_uri
   ```

## Usage

### Basic Example

```python
from src.embedding import EmbeddingModel
from src.vectordb import VectorStore
from src.llm import LLMModel
from src.graph import RAGGraph
from langchain_core.documents import Document

# Initialize components
embedding_model = EmbeddingModel()
vector_store = VectorStore(
    embedding=embedding_model.get_embeddings(),
    collection_name="my_collection"
)

# Create sample documents
docs = [
    Document(page_content="Your document text here...", metadata={"source": "doc1"})
]
vector_store.create_from_documents(docs)

# Initialize LLM and RAG graph
llm = LLMModel()
rag_graph = RAGGraph(
    vector_store=vector_store,
    llm=llm,
    max_rewrites=2,
    relevance_threshold=0.5
)

# Ask a question
result = rag_graph.invoke("What is this document about?")

print(f"Answer: {result['answer']}")
print(f"Steps taken: {result['steps']}")
print(f"Query rewrites: {result['rewrites']}")
print(f"Answer grounded: {result['grounded']}")
```

### Using with Document Ingestion

```python
from src.processing import collect_documents_paths, convert_to_markdown, chunk
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

# Collect and process documents
paths = collect_documents_paths("path/to/documents")
iter_document = convert_to_markdown(paths)
chunker = HybridChunker()
chunk_iter = chunk(iter_document, chunker)
chunks = list(chunk_iter)

# Create vector store with documents
vector_store.create_from_documents(chunks)

# Now use with RAG graph as above
```

### Streaming Mode

For real-time progress updates:

```python
for step_output in rag_graph.stream("Your question here"):
    for node, state in step_output.items():
        print(f"Processing node: {node}")
        if "steps" in state and state["steps"]:
            print(f"  {state['steps'][-1]}")
```

## Running Examples

The project includes three example modes:

### 1. Basic Mode
Tests the RAG graph with simple sample documents:
```bash
python examples/langgraph_rag_example.py --mode basic
```

### 2. Streaming Mode
Shows step-by-step execution:
```bash
python examples/langgraph_rag_example.py --mode streaming
```

### 3. Advanced Mode
Interactive Q&A with real documents:
```bash
python examples/langgraph_rag_example.py --mode advanced --docs-path /path/to/your/documents
```

## Configuration Options

### RAGGraph Parameters

- **vector_store** (VectorStore): Initialized vector store for document retrieval
- **llm** (LLM): Initialized LLM instance
- **max_rewrites** (int, default=2): Maximum number of query rewrites
- **relevance_threshold** (float, default=0.5): Minimum relevance score for documents (0-1)

### State Object

The graph maintains the following state:

```python
{
    "question": str,              # Current question (may be rewritten)
    "generation": str,            # Generated answer
    "documents": list[Document],  # Retrieved documents
    "steps": list[str],          # Processing steps taken
    "rewrite_count": int,        # Number of rewrites performed
    "relevance_scores": list[float],  # Document relevance scores
    "answer_grounded": bool      # Whether answer is grounded in docs
}
```

## Graph Nodes

| Node | Description |
|------|-------------|
| `analyze_query` | Analyzes and potentially expands the user's query |
| `retrieve` | Retrieves documents from vector store |
| `grade_documents` | Grades relevance of retrieved documents |
| `generate` | Generates answer from relevant documents |
| `check_hallucination` | Verifies answer is grounded in documents |
| `rewrite_query` | Rewrites query for better retrieval |

## Comparison: Traditional RAG vs LangGraph RAG

| Feature | Traditional RAG | LangGraph RAG |
|---------|----------------|---------------|
| Document Retrieval | ✅ | ✅ |
| Answer Generation | ✅ | ✅ |
| Relevance Grading | ❌ | ✅ |
| Query Rewriting | ❌ | ✅ |
| Hallucination Detection | ❌ | ✅ |
| Self-Correction | ❌ | ✅ |
| Stateful Workflow | ❌ | ✅ |
| Streaming Support | Limited | ✅ |

## Best Practices

1. **Set appropriate max_rewrites**: Start with 2-3 rewrites. Too many can be slow.

2. **Tune relevance_threshold**: Higher values (0.6-0.8) are stricter but may miss relevant docs. Lower values (0.3-0.5) are more permissive.

3. **Monitor token usage**: LangGraph makes multiple LLM calls for grading and checking. Be mindful of API costs.

4. **Use streaming for long processes**: Provides better UX by showing progress.

5. **Customize prompts**: The prompts in `src/graph.py` can be customized for your specific use case.

## Troubleshooting

### "No relevant documents found after grading"
- Lower the `relevance_threshold`
- Ensure your documents cover the query topic
- Check vector store has documents indexed

### "Max rewrites reached without grounded answer"
- Increase `max_rewrites`
- Review your document quality and coverage
- Check if LLM is hallucinating despite grading

### Slow performance
- Reduce `max_rewrites`
- Use smaller/faster LLM for grading steps
- Implement caching for repeated queries

## Advanced Customization

### Adding Custom Nodes

```python
def custom_node(state: GraphState) -> GraphState:
    # Your custom logic
    return {
        **state,
        "steps": ["Custom processing step"]
    }

# In _build_graph()
workflow.add_node("custom_node", custom_node)
workflow.add_edge("analyze_query", "custom_node")
```

### Custom Conditional Logic

```python
def custom_decision(state: GraphState) -> Literal["path_a", "path_b"]:
    # Your decision logic
    if some_condition:
        return "path_a"
    return "path_b"

workflow.add_conditional_edges(
    "your_node",
    custom_decision,
    {
        "path_a": "node_a",
        "path_b": "node_b"
    }
)
```

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)

## Future Enhancements

Potential improvements to the LangGraph integration:

- [ ] Multi-hop reasoning for complex questions
- [ ] Parallel retrieval from multiple sources
- [ ] Agent-based document analysis
- [ ] Adaptive retrieval strategies
- [ ] Conversation memory and context
- [ ] Tool calling for external APIs
- [ ] Multi-modal document processing

## Contributing

To extend the LangGraph functionality:

1. Modify `src/graph.py` to add new nodes or edges
2. Update the example scripts in `examples/`
3. Add tests in `src/test/`
4. Update this documentation

## License

Same license as the main project.