-- RAG Logs Database Schema for SQLite
-- Execute this file to create the database structure

-- Table for storing different RAG configurations
CREATE TABLE IF NOT EXISTS configurations (
    config_id TEXT PRIMARY KEY,
    llm_provider TEXT NOT NULL,
    llm_model TEXT NOT NULL,
    temperature REAL NOT NULL,
    embedding_provider TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_device TEXT NOT NULL,
    vector_db TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    chunk_overlap INTEGER NOT NULL,
    top_k INTEGER NOT NULL,
    enhanced_processing BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    config_hash TEXT UNIQUE NOT NULL
);

-- Table for storing document metadata
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT,
    file_hash TEXT UNIQUE NOT NULL,
    file_size_bytes INTEGER,
    total_pages INTEGER,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    processing_status TEXT DEFAULT 'pending'
);

-- Table for chat sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    config_id TEXT NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_name TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (config_id) REFERENCES configurations(config_id)
);

-- Junction table for documents in sessions (many-to-many)
CREATE TABLE IF NOT EXISTS document_sessions (
    document_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (document_id, session_id),
    FOREIGN KEY (document_id) REFERENCES documents(document_id),
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
);

-- Table for user queries
CREATE TABLE IF NOT EXISTS queries (
    query_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    config_id TEXT NOT NULL,
    user_question TEXT NOT NULL,
    asked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    response_time_seconds REAL,
    retrieved_chunks_count INTEGER,
    query_hash TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id),
    FOREIGN KEY (config_id) REFERENCES configurations(config_id)
);

-- Table for model responses
CREATE TABLE IF NOT EXISTS responses (
    response_id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,
    model_answer TEXT NOT NULL,
    confidence_score REAL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_response TEXT, -- JSON as TEXT in SQLite
    response_hash TEXT NOT NULL,
    FOREIGN KEY (query_id) REFERENCES queries(query_id)
);

-- Table for retrieved chunks
CREATE TABLE IF NOT EXISTS retrieved_chunks (
    chunk_id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,
    document_id TEXT,
    chunk_content TEXT NOT NULL,
    chunk_index INTEGER,
    page_number INTEGER,
    similarity_score REAL NOT NULL,
    relevance_rank INTEGER NOT NULL,
    relevance_percentage REAL,
    metadata TEXT, -- JSON as TEXT in SQLite
    chunk_hash TEXT NOT NULL,
    FOREIGN KEY (query_id) REFERENCES queries(query_id),
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

-- Table for caching embeddings
CREATE TABLE IF NOT EXISTS embeddings_cache (
    embedding_id TEXT PRIMARY KEY,
    content_hash TEXT UNIQUE NOT NULL,
    embedding_provider TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_vector BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    vector_dimension INTEGER
);

-- Table for performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    metric_id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,
    embedding_time_seconds REAL,
    retrieval_time_seconds REAL,
    llm_response_time_seconds REAL,
    total_time_seconds REAL,
    memory_usage_mb INTEGER,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (query_id) REFERENCES queries(query_id)
);

-- Table for user feedback
CREATE TABLE IF NOT EXISTS user_feedback (
    feedback_id TEXT PRIMARY KEY,
    response_id TEXT NOT NULL,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    helpful BOOLEAN,
    tags TEXT, -- JSON as TEXT in SQLite
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT,
    FOREIGN KEY (response_id) REFERENCES responses(response_id)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_configurations_hash ON configurations(config_hash);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_queries_session ON queries(session_id);
CREATE INDEX IF NOT EXISTS idx_queries_config ON queries(config_id);
CREATE INDEX IF NOT EXISTS idx_queries_hash ON queries(query_hash);
CREATE INDEX IF NOT EXISTS idx_responses_query ON responses(query_id);
CREATE INDEX IF NOT EXISTS idx_chunks_query ON retrieved_chunks(query_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON retrieved_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_performance_query ON performance_metrics(query_id);
CREATE INDEX IF NOT EXISTS idx_feedback_response ON user_feedback(response_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_hash ON embeddings_cache(content_hash);

-- Views for common queries
CREATE VIEW IF NOT EXISTS query_analysis AS
SELECT
    q.query_id,
    q.user_question,
    q.asked_at,
    c.llm_provider,
    c.llm_model,
    c.embedding_provider,
    c.embedding_model,
    c.chunk_size,
    c.top_k,
    r.model_answer,
    q.response_time_seconds,
    q.retrieved_chunks_count,
    COALESCE(AVG(uf.rating), 0) as avg_rating
FROM queries q
JOIN configurations c ON q.config_id = c.config_id
JOIN responses r ON q.query_id = r.query_id
LEFT JOIN user_feedback uf ON r.response_id = uf.response_id
GROUP BY q.query_id;

CREATE VIEW IF NOT EXISTS config_performance AS
SELECT
    c.config_id,
    c.llm_provider,
    c.llm_model,
    c.embedding_provider,
    c.embedding_model,
    c.chunk_size,
    c.top_k,
    COUNT(q.query_id) as total_queries,
    AVG(q.response_time_seconds) as avg_response_time,
    AVG(COALESCE(uf.rating, 0)) as avg_rating
FROM configurations c
LEFT JOIN queries q ON c.config_id = q.config_id
LEFT JOIN responses r ON q.query_id = r.query_id
LEFT JOIN user_feedback uf ON r.response_id = uf.response_id
GROUP BY c.config_id;