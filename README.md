# Deep Learning RAG Application

A comprehensive Retrieval-Augmented Generation (RAG) system with multiple vector database support, performance monitoring, and analytics capabilities.

## 🚀 Features

- **Multiple LLM Providers**: Groq, Cerebras/OpenAI support
- **Multiple Vector Databases**: FAISS, Qdrant, Weaviate, ChromaDB, Milvus
- **Multiple Embedding Models**: HuggingFace, OpenAI, Cohere
- **Performance Monitoring**: SQLite-based logging and Grafana dashboards
- **Document Processing**: PDF, text file support with advanced chunking
- **Web Interface**: Streamlit-based user interface

## 📋 Prerequisites

- Python 3.8+
- Docker and Docker Compose (for vector databases)
- SQLite (for logging)
- Make (for convenience commands)

## ⚡ Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd deep_learning
make install
```

### 2. Setup Environment
Create a `.env` file:
```bash
# LLM Configuration
LLM_PROVIDER=groq  # groq or openai
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key

# Embedding Configuration
EMBEDDING_PROVIDER=huggingface  # huggingface, openai, or cohere
EMBEDDING_MODEL=all-MiniLM-L6-v2
COHERE_API_KEY=your_cohere_api_key

# Vector Database
VECTOR_DB=faiss  # faiss, qdrant, weaviate, chroma, or milvus

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=4
TEMPERATURE=0.1
```

### 3. Initialize Database
```bash
make setup-db
```

### 4. Start Vector Database (if not using FAISS)
```bash
# For development (FAISS - no server needed)
make start-faiss

# For production (choose one)
make start-qdrant
# make start-weaviate
# make start-chroma
# make start-milvus
```

### 5. Run Application
```bash
make run
```

Visit http://localhost:8501 to access the Streamlit interface.

## 🛠️ Available Commands

### Application
- `make install` - Install Python dependencies
- `make run` - Start the Streamlit application
- `make clean` - Clean up temporary files

### Vector Databases
- `make start-faiss` - Set VECTOR_DB to faiss (no server needed)
- `make start-qdrant` - Start Qdrant vector database
- `make start-weaviate` - Start Weaviate vector database
- `make start-chroma` - Start ChromaDB vector database
- `make start-milvus` - Start Milvus vector database

### Database Management
- `make setup-db` - Initialize/setup the RAG logs database
- `make stop-qdrant` - Stop Qdrant
- `make stop-weaviate` - Stop Weaviate
- `make stop-chroma` - Stop ChromaDB
- `make stop-milvus` - Stop Milvus
- `make stop-all` - Stop all vector databases

### Monitoring
- `make status` - Show status of all services
- `make health-check` - Check health of all running services
- `make logs-*` - Show logs for specific services

### Data Management
- `make clean-data` - Remove all vector database data
- `make backup-data` - Backup vector database data

### Quick Setup
- `make setup-dev` - Setup development environment with FAISS
- `make setup-prod` - Setup production environment with Qdrant

## 📊 Performance Monitoring

The application includes comprehensive performance monitoring:

### Database Schema
- **Configurations**: Track different RAG parameter combinations
- **Documents**: Document metadata and processing status
- **Chat Sessions**: User interaction sessions
- **Queries & Responses**: Question-answer pairs with performance metrics
- **Retrieved Chunks**: Document chunks used for context
- **User Feedback**: Ratings and feedback on responses
- **Performance Metrics**: Detailed timing and resource usage

### Analytics Views
- **Query Analysis**: Combined view of queries with configuration and performance data
- **Config Performance**: Aggregated performance metrics by configuration

### Grafana Dashboard
For advanced monitoring, see [README_GRAFANA.md](README_GRAFANA.md) for Grafana setup instructions.

## 🗂️ Project Structure

```
deep_learning/
├── interface.py              # Main Streamlit application
├── requirements.txt          # Python dependencies
├── Makefile                 # Convenient commands
├── .env                     # Environment configuration
├── db_schema.sql            # Database schema
├── rag_logs.db             # SQLite performance database
├── docker/                 # Docker configurations
│   ├── qdrant/
│   ├── weaviate/
│   ├── chroma/
│   ├── milvus/
│   └── grafana/
└── README_GRAFANA.md       # Grafana setup instructions
```

## 🔧 Configuration

### LLM Providers
- **Groq**: Fast inference with Llama models
- **OpenAI/Cerebras**: GPT models and alternatives

### Vector Databases
- **FAISS**: Local, CPU-based (development)
- **Qdrant**: High-performance vector search (recommended for production)
- **Weaviate**: GraphQL-based vector database
- **ChromaDB**: Simple, local vector database
- **Milvus**: Scalable vector database

### Embedding Models
- **HuggingFace**: sentence-transformers models (all-MiniLM-L6-v2, etc.)
- **OpenAI**: text-embedding-ada-002
- **Cohere**: embed-english-v2.0

## 📈 Performance Features

### Automatic Logging
- Query response times
- Chunk retrieval metrics
- LLM generation timing
- Memory usage tracking
- User satisfaction ratings

### Optimizations
- Embedding caching
- Query result caching
- Parallel document processing
- Configurable chunk sizes
- Enhanced text processing

## 🚧 Troubleshooting

### Common Issues

**Dependencies**
```bash
# Update requirements
pip install -r requirements.txt --upgrade
```

**Vector Database Connection**
```bash
# Check service status
make status

# View logs
make logs-qdrant  # or other service
```

**Database Issues**
```bash
# Reset database
rm rag_logs.db
make setup-db
```

**Port Conflicts**
- Streamlit: 8501
- Qdrant: 6333
- Weaviate: 8080
- ChromaDB: 8000
- Milvus: 19530
- Grafana: 3000

### Performance Tips

1. **Choose appropriate chunk size** (500-1500 tokens)
2. **Adjust top_k** based on document complexity
3. **Use FAISS for development**, vector DB for production
4. **Monitor response times** through the analytics dashboard
5. **Enable caching** for repeated queries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs using `make logs-*` commands
3. Check service status with `make status`
4. Open an issue on GitHub

---

**Happy RAG-ing!** 🚀