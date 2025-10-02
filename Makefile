# Deep Learning RAG Application Makefile
# Convenient commands for managing vector databases and the application

.PHONY: help install run clean logs status stop-all

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "Application:"
	@echo "  make install     - Install Python dependencies"
	@echo "  make run         - Start the Streamlit application"
	@echo "  make clean       - Clean up temporary files and volumes"
	@echo ""
	@echo "Vector Databases:"
	@echo "  make start-faiss     - Set VECTOR_DB to faiss (no server needed)"
	@echo "  make start-qdrant    - Start Qdrant vector database"
	@echo "  make start-weaviate  - Start Weaviate vector database"
	@echo "  make start-chroma    - Start ChromaDB vector database"
	@echo "  make start-milvus    - Start Milvus vector database"
	@echo ""
	@echo "Database Management:"
	@echo "  make stop-qdrant     - Stop Qdrant"
	@echo "  make stop-weaviate   - Stop Weaviate"
	@echo "  make stop-chroma     - Stop ChromaDB"
	@echo "  make stop-milvus     - Stop Milvus"
	@echo "  make stop-all        - Stop all vector databases"
	@echo ""
	@echo "Monitoring:"
	@echo "  make status          - Show status of all services"
	@echo "  make logs-qdrant     - Show Qdrant logs"
	@echo "  make logs-weaviate   - Show Weaviate logs"
	@echo "  make logs-chroma     - Show ChromaDB logs"
	@echo "  make logs-milvus     - Show Milvus logs"
	@echo ""
	@echo "Data Management:"
	@echo "  make clean-data      - Remove all vector database data"
	@echo "  make backup-data     - Backup vector database data"

# Application commands
install:
	pip install -r req.txt

run:
	streamlit run interface.py

# Vector Database startup commands
start-faiss:
	@echo "Setting VECTOR_DB=faiss in .env file..."
	@sed -i 's/^VECTOR_DB=.*/VECTOR_DB=faiss/' .env || echo "VECTOR_DB=faiss" >> .env
	@echo "FAISS is ready to use (no server required)"
	@echo "Run 'make run' to start the application"

start-qdrant:
	@echo "Starting Qdrant vector database..."
	@sed -i 's/^VECTOR_DB=.*/VECTOR_DB=qdrant/' .env || echo "VECTOR_DB=qdrant" >> .env
	docker-compose -f docker/qdrant/docker-compose.yml up -d
	@echo "Qdrant started at http://localhost:6333"
	@echo "Run 'make run' to start the application"

start-weaviate:
	@echo "Starting Weaviate vector database..."
	@sed -i 's/^VECTOR_DB=.*/VECTOR_DB=weaviate/' .env || echo "VECTOR_DB=weaviate" >> .env
	docker-compose -f docker/weaviate/docker-compose.yml up -d
	@echo "Weaviate started at http://localhost:8080"
	@echo "Run 'make run' to start the application"

start-chroma:
	@echo "Starting ChromaDB vector database..."
	@sed -i 's/^VECTOR_DB=.*/VECTOR_DB=chroma/' .env || echo "VECTOR_DB=chroma" >> .env
	docker-compose -f docker/chroma/docker-compose.yml up -d
	@echo "ChromaDB started at http://localhost:8000"
	@echo "Run 'make run' to start the application"

start-milvus:
	@echo "Starting Milvus vector database..."
	@sed -i 's/^VECTOR_DB=.*/VECTOR_DB=milvus/' .env || echo "VECTOR_DB=milvus" >> .env
	docker-compose -f docker/milvus/docker-compose.yml up -d
	@echo "Milvus started at http://localhost:19530"
	@echo "Run 'make run' to start the application"

# Stop commands
stop-qdrant:
	docker-compose -f docker/qdrant/docker-compose.yml down

stop-weaviate:
	docker-compose -f docker/weaviate/docker-compose.yml down

stop-chroma:
	docker-compose -f docker/chroma/docker-compose.yml down

stop-milvus:
	docker-compose -f docker/milvus/docker-compose.yml down

stop-all:
	@echo "Stopping all vector databases..."
	-docker-compose -f docker/qdrant/docker-compose.yml down
	-docker-compose -f docker/weaviate/docker-compose.yml down
	-docker-compose -f docker/chroma/docker-compose.yml down
	-docker-compose -f docker/milvus/docker-compose.yml down
	@echo "All services stopped"

# Status and monitoring
status:
	@echo "=== Vector Database Status ==="
	@echo ""
	@echo "Current VECTOR_DB setting:"
	@grep "^VECTOR_DB=" .env || echo "VECTOR_DB not set"
	@echo ""
	@echo "Qdrant:"
	@docker-compose -f docker/qdrant/docker-compose.yml ps 2>/dev/null || echo "  Not running"
	@echo ""
	@echo "Weaviate:"
	@docker-compose -f docker/weaviate/docker-compose.yml ps 2>/dev/null || echo "  Not running"
	@echo ""
	@echo "ChromaDB:"
	@docker-compose -f docker/chroma/docker-compose.yml ps 2>/dev/null || echo "  Not running"
	@echo ""
	@echo "Milvus:"
	@docker-compose -f docker/milvus/docker-compose.yml ps 2>/dev/null || echo "  Not running"

logs-qdrant:
	docker-compose -f docker/qdrant/docker-compose.yml logs -f

logs-weaviate:
	docker-compose -f docker/weaviate/docker-compose.yml logs -f

logs-chroma:
	docker-compose -f docker/chroma/docker-compose.yml logs -f

logs-milvus:
	docker-compose -f docker/milvus/docker-compose.yml logs -f

# Data management
clean-data:
	@echo "WARNING: This will delete all vector database data!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	@echo "Stopping all services..."
	@make stop-all
	@echo "Removing data volumes..."
	-sudo rm -rf volumes/
	-sudo rm -rf chroma_db/
	@echo "Data cleaned"

backup-data:
	@echo "Creating backup of vector database data..."
	@mkdir -p backups
	@timestamp=$$(date +%Y%m%d_%H%M%S) && \
	tar -czf "backups/vectordb_backup_$$timestamp.tar.gz" volumes/ chroma_db/ 2>/dev/null || true
	@echo "Backup created in backups/ directory"

# Development commands
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name ".DS_Store" -delete
	find . -type f -name "*.tmp" -delete
	@echo "Temporary files cleaned"

# Health checks
health-check:
	@echo "=== Health Check ==="
	@echo "Checking Qdrant..."
	@curl -s http://localhost:6333/health >/dev/null && echo "✓ Qdrant: Healthy" || echo "✗ Qdrant: Not responding"
	@echo "Checking Weaviate..."
	@curl -s http://localhost:8080/v1/.well-known/ready >/dev/null && echo "✓ Weaviate: Healthy" || echo "✗ Weaviate: Not responding"
	@echo "Checking ChromaDB..."
	@curl -s http://localhost:8000/api/v1/heartbeat >/dev/null && echo "✓ ChromaDB: Healthy" || echo "✗ ChromaDB: Not responding"
	@echo "Checking Milvus..."
	@curl -s http://localhost:9091/healthz >/dev/null && echo "✓ Milvus: Healthy" || echo "✗ Milvus: Not responding"

# Quick setup commands
setup-dev:
	@echo "Setting up development environment..."
	make install
	make start-faiss
	@echo "Development environment ready!"
	@echo "Run 'make run' to start the application"

setup-prod:
	@echo "Setting up production environment with Qdrant..."
	make install
	make start-qdrant
	@echo "Production environment ready!"
	@echo "Run 'make run' to start the application"