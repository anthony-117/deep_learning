# Vector Database Setup Guide

This guide explains how to configure different vector databases for the RAG system.

## Quick Start

1. Set the `VECTOR_DB` environment variable in your `.env` file
2. Install the required dependencies
3. Configure any additional environment variables (if needed)

## Supported Vector Databases

### 1. FAISS (Default)
**No additional setup required** - works out of the box.

```bash
# .env
VECTOR_DB=faiss
```

**Dependencies:**
```bash
pip install faiss-cpu
```

---

### 2. Chroma
Local vector database with persistence.

```bash
# .env
VECTOR_DB=chroma
```

**Dependencies:**
```bash
pip install chromadb
```

**Setup Options:**

**Option A: Local Storage (Default)**
- Local storage in `./chroma_db` directory
- No additional configuration needed
- Persistent across sessions

**Option B: Docker Instance**
```bash
# Using docker-compose
docker-compose -f docker-compose.chroma.yml up -d
```

---

### 3. Qdrant
High-performance vector search engine.

```bash
# .env
VECTOR_DB=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key_here  # Optional for local instances
```

**Dependencies:**
```bash
pip install qdrant-client
```

**Setup Options:**

**Option A: Local Docker Instance**
```bash
# Using docker run
docker run -p 6333:6333 qdrant/qdrant

# Using docker-compose (recommended)
docker-compose -f docker-compose.qdrant.yml up -d
```

**Option B: Qdrant Cloud**
1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a cluster
3. Get your URL and API key
4. Set `QDRANT_URL` and `QDRANT_API_KEY`

---

### 4. Pinecone
Managed vector database service.

```bash
# .env
VECTOR_DB=pinecone
PINECONE_API_KEY=your_api_key_here
PINECONE_ENVIRONMENT=your_environment_here
```

**Dependencies:**
```bash
pip install pinecone-client
```

**Setup:**
1. Sign up at [pinecone.io](https://pinecone.io)
2. Create a project
3. Get your API key from the dashboard
4. Note your environment (e.g., `us-west1-gcp`)
5. Set both `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT`

---

### 5. Weaviate
Open-source vector database.

```bash
# .env
VECTOR_DB=weaviate
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your_api_key_here  # Optional for local instances
```

**Dependencies:**
```bash
pip install weaviate-client
```

**Setup Options:**

**Option A: Local Docker Instance**
```bash
# Using docker run
docker run -p 8080:8080 semitechnologies/weaviate:latest

# Using docker-compose (recommended)
docker-compose -f docker-compose.weaviate.yml up -d
```

**Option B: Weaviate Cloud Services (WCS)**
1. Sign up at [console.weaviate.cloud](https://console.weaviate.cloud)
2. Create a cluster
3. Get your cluster URL and API key
4. Set `WEAVIATE_URL` and `WEAVIATE_API_KEY`

---

### 6. Milvus
Scalable vector database for AI applications.

```bash
# .env
VECTOR_DB=milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

**Dependencies:**
```bash
pip install pymilvus
```

**Setup Options:**

**Option A: Local Docker Instance**
```bash
# Using provided docker-compose file
docker-compose -f docker-compose.milvus.yml up -d
```

**Option B: Zilliz Cloud (Managed Milvus)**
1. Sign up at [zilliz.com](https://zilliz.com)
2. Create a cluster
3. Get connection details
4. Update `MILVUS_HOST` and `MILVUS_PORT`

---

## Environment Variable Reference

| Variable | Required For | Default | Description |
|----------|-------------|---------|-------------|
| `VECTOR_DB` | All | `faiss` | Vector database to use |
| `QDRANT_URL` | Qdrant | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | Qdrant | - | Qdrant API key (optional for local) |
| `PINECONE_API_KEY` | Pinecone | - | Pinecone API key (required) |
| `PINECONE_ENVIRONMENT` | Pinecone | - | Pinecone environment (required) |
| `WEAVIATE_URL` | Weaviate | `http://localhost:8080` | Weaviate server URL |
| `WEAVIATE_API_KEY` | Weaviate | - | Weaviate API key (optional for local) |
| `MILVUS_HOST` | Milvus | `localhost` | Milvus server host |
| `MILVUS_PORT` | Milvus | `19530` | Milvus server port |

## Example .env File

```bash
# === LLM API KEYS ===
GROQ_API_KEY="your_groq_api_key_here"

# === VECTOR DATABASE CONFIGURATION ===
VECTOR_DB=chroma

# === VECTOR DB SPECIFIC CONFIGURATIONS ===
# Uncomment and configure as needed:

# Qdrant
# QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=your_qdrant_api_key

# Pinecone
# PINECONE_API_KEY=your_pinecone_api_key
# PINECONE_ENVIRONMENT=us-west1-gcp

# Weaviate
# WEAVIATE_URL=http://localhost:8080
# WEAVIATE_API_KEY=your_weaviate_api_key

# Milvus
# MILVUS_HOST=localhost
# MILVUS_PORT=19530
```

## Performance Comparison

| Database | Best For | Pros | Cons |
|----------|----------|------|------|
| **FAISS** | Quick testing, local development | Fast, no setup | Not persistent, single-node |
| **Chroma** | Small to medium projects | Easy setup, persistent | Limited scalability |
| **Qdrant** | Production apps | High performance, good APIs | Requires server setup |
| **Pinecone** | Managed solutions | Fully managed, scalable | Cost, vendor lock-in |
| **Weaviate** | Complex queries | Rich querying, GraphQL | More complex setup |
| **Milvus** | Large scale | Highly scalable, enterprise-ready | Complex deployment |

## Docker Setup Commands

Individual Docker Compose files are organized in the `docker/` directory. Start only the one you need:

```bash
# Start Qdrant
docker-compose -f docker/qdrant/docker-compose.yml up -d

# Start Weaviate
docker-compose -f docker/weaviate/docker-compose.yml up -d

# Start ChromaDB
docker-compose -f docker/chroma/docker-compose.yml up -d

# Start Milvus
docker-compose -f docker/milvus/docker-compose.yml up -d

# Stop any running service
docker-compose -f docker/{service}/docker-compose.yml down

# View logs
docker-compose -f docker/{service}/docker-compose.yml logs -f

# Check status
docker-compose -f docker/{service}/docker-compose.yml ps
```

## Makefile Commands

For convenience, use the provided Makefile:

```bash
# Quick setup and start
make start-qdrant    # Start Qdrant and set VECTOR_DB=qdrant
make start-weaviate  # Start Weaviate and set VECTOR_DB=weaviate
make start-chroma    # Start ChromaDB and set VECTOR_DB=chroma
make start-milvus    # Start Milvus and set VECTOR_DB=milvus
make start-faiss     # Set VECTOR_DB=faiss (no server needed)

# Application management
make install         # Install dependencies
make run            # Start Streamlit application
make status         # Show all service status
make health-check   # Check service health

# Stop services
make stop-all       # Stop all vector databases
make stop-qdrant    # Stop specific service

# Development shortcuts
make setup-dev      # Install deps + start FAISS
make setup-prod     # Install deps + start Qdrant
```

**Data Persistence:**
- All data is stored in `./volumes/{service}` directories
- Data persists between container restarts
- To reset data, stop the service and delete the volume directory

## Troubleshooting

### Common Issues

1. **Import Errors**: Install the required dependencies for your chosen vector DB
2. **Connection Errors**: Check if the vector database server is running
3. **Authentication Errors**: Verify API keys and credentials
4. **Permission Errors**: Ensure proper file/directory permissions for local databases

### Getting Help

- Check the official documentation for each vector database
- Ensure environment variables are properly set in your `.env` file
- Verify network connectivity for remote databases
- Check firewall settings for local Docker instances