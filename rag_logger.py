import sqlite3
import json
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import os


class RAGLogger:
    """Logger for RAG system interactions and configurations"""

    def __init__(self, db_path: str = "rag_logs.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database with schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Read and execute schema
            schema_path = os.path.join(os.path.dirname(__file__), "db_schema.sql")
            if os.path.exists(schema_path):
                with open(schema_path, 'r') as f:
                    conn.executescript(f.read())
            else:
                # Fallback: create basic schema if file not found
                self._create_basic_schema(conn)

    def _create_basic_schema(self, conn):
        """Fallback schema creation"""
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS configurations (
                config_id TEXT PRIMARY KEY,
                llm_provider TEXT, llm_model TEXT, temperature REAL,
                embedding_provider TEXT, embedding_model TEXT, embedding_device TEXT,
                vector_db TEXT, chunk_size INTEGER, chunk_overlap INTEGER, top_k INTEGER,
                enhanced_processing BOOLEAN, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                config_hash TEXT UNIQUE
            );
            CREATE TABLE IF NOT EXISTS queries (
                query_id TEXT PRIMARY KEY, session_id TEXT, config_id TEXT,
                user_question TEXT, asked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                response_time_seconds REAL, retrieved_chunks_count INTEGER, query_hash TEXT
            );
            CREATE TABLE IF NOT EXISTS responses (
                response_id TEXT PRIMARY KEY, query_id TEXT, model_answer TEXT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, response_hash TEXT
            );
            CREATE TABLE IF NOT EXISTS retrieved_chunks (
                chunk_id TEXT PRIMARY KEY, query_id TEXT, chunk_content TEXT,
                similarity_score REAL, relevance_rank INTEGER, relevance_percentage REAL,
                metadata TEXT, chunk_hash TEXT
            );
        """)

    def _generate_hash(self, data: Any) -> str:
        """Generate MD5 hash for data"""
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True)
        elif not isinstance(data, str):
            data = str(data)
        return hashlib.md5(data.encode()).hexdigest()

    def log_configuration(self, config: Dict[str, Any]) -> str:
        """Log a RAG configuration and return config_id"""
        config_id = str(uuid.uuid4())
        config_hash = self._generate_hash(config)

        with sqlite3.connect(self.db_path) as conn:
            # Check if configuration already exists
            cursor = conn.execute(
                "SELECT config_id FROM configurations WHERE config_hash = ?",
                (config_hash,)
            )
            existing = cursor.fetchone()
            if existing:
                return existing[0]

            # Insert new configuration
            conn.execute("""
                INSERT INTO configurations (
                    config_id, llm_provider, llm_model, temperature,
                    embedding_provider, embedding_model, embedding_device,
                    vector_db, chunk_size, chunk_overlap, top_k,
                    enhanced_processing, config_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                config_id,
                config.get('llm_provider'),
                config.get('model'),  # Note: 'model' in your config, 'llm_model' in DB
                config.get('temp'),
                config.get('embedding_provider'),
                config.get('embedding_model'),
                config.get('embedding_device'),
                config.get('vector_db'),
                config.get('chunk_size'),
                config.get('overlap'),  # Note: 'overlap' in config, 'chunk_overlap' in DB
                config.get('top_k'),
                config.get('enhanced', False),
                config_hash
            ))

        return config_id

    def log_query_response(
        self,
        session_id: str,
        config_id: str,
        question: str,
        answer: str,
        retrieved_chunks: List[Dict],
        response_time: float = None
    ) -> str:
        """Log a complete query-response interaction"""

        query_id = str(uuid.uuid4())
        response_id = str(uuid.uuid4())
        query_hash = self._generate_hash(question)
        response_hash = self._generate_hash(answer)

        with sqlite3.connect(self.db_path) as conn:
            # Log query
            conn.execute("""
                INSERT INTO queries (
                    query_id, session_id, config_id, user_question,
                    response_time_seconds, retrieved_chunks_count, query_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                query_id, session_id, config_id, question,
                response_time, len(retrieved_chunks), query_hash
            ))

            # Log response
            conn.execute("""
                INSERT INTO responses (
                    response_id, query_id, model_answer, response_hash
                ) VALUES (?, ?, ?, ?)
            """, (response_id, query_id, answer, response_hash))

            # Log retrieved chunks
            for chunk in retrieved_chunks:
                chunk_id = str(uuid.uuid4())
                chunk_hash = self._generate_hash(chunk.get('content', ''))

                conn.execute("""
                    INSERT INTO retrieved_chunks (
                        chunk_id, query_id, chunk_content, similarity_score,
                        relevance_rank, relevance_percentage, metadata, chunk_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk_id, query_id,
                    chunk.get('content', ''),
                    chunk.get('similarity_score'),
                    chunk.get('rank'),
                    chunk.get('relevance_percentage'),
                    json.dumps(chunk.get('metadata', {})),
                    chunk_hash
                ))

        return query_id

    def create_session(self, config_id: str, session_name: str = None) -> str:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO chat_sessions (session_id, config_id, session_name)
                VALUES (?, ?, ?)
            """, (session_id, config_id, session_name))

        return session_id

    def log_feedback(self, response_id: str, rating: int, feedback_text: str = None, helpful: bool = None):
        """Log user feedback for a response"""
        feedback_id = str(uuid.uuid4())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO user_feedback (
                    feedback_id, response_id, rating, feedback_text, helpful
                ) VALUES (?, ?, ?, ?, ?)
            """, (feedback_id, response_id, rating, feedback_text, helpful))

    def get_config_comparison(self, limit: int = 10) -> List[Dict]:
        """Get configuration performance comparison"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT
                    c.config_id, c.llm_provider, c.llm_model,
                    c.embedding_provider, c.embedding_model,
                    c.chunk_size, c.top_k,
                    COUNT(q.query_id) as total_queries,
                    AVG(q.response_time_seconds) as avg_response_time,
                    AVG(COALESCE(uf.rating, 0)) as avg_rating
                FROM configurations c
                LEFT JOIN queries q ON c.config_id = q.config_id
                LEFT JOIN responses r ON q.query_id = r.query_id
                LEFT JOIN user_feedback uf ON r.response_id = uf.response_id
                GROUP BY c.config_id
                ORDER BY total_queries DESC, avg_rating DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_similar_queries(self, question: str, limit: int = 5) -> List[Dict]:
        """Find similar questions that have been asked before"""
        query_hash = self._generate_hash(question)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT q.user_question, r.model_answer, q.asked_at,
                       c.llm_provider, c.llm_model
                FROM queries q
                JOIN responses r ON q.query_id = r.query_id
                JOIN configurations c ON q.config_id = c.config_id
                WHERE q.query_hash = ? OR q.user_question LIKE ?
                ORDER BY q.asked_at DESC
                LIMIT ?
            """, (query_hash, f"%{question[:50]}%", limit))

            return [dict(row) for row in cursor.fetchall()]

    def get_query_history(self, session_id: str = None, limit: int = 50) -> List[Dict]:
        """Get query history for analysis"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if session_id:
                cursor = conn.execute("""
                    SELECT q.user_question, r.model_answer, q.asked_at,
                           q.response_time_seconds, q.retrieved_chunks_count
                    FROM queries q
                    JOIN responses r ON q.query_id = r.query_id
                    WHERE q.session_id = ?
                    ORDER BY q.asked_at DESC
                    LIMIT ?
                """, (session_id, limit))
            else:
                cursor = conn.execute("""
                    SELECT q.user_question, r.model_answer, q.asked_at,
                           q.response_time_seconds, q.retrieved_chunks_count,
                           c.llm_provider, c.llm_model
                    FROM queries q
                    JOIN responses r ON q.query_id = r.query_id
                    JOIN configurations c ON q.config_id = c.config_id
                    ORDER BY q.asked_at DESC
                    LIMIT ?
                """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def export_data(self, output_file: str = None, format: str = "json"):
        """Export all data for analysis"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"rag_export_{timestamp}.{format}"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Export query analysis view
            cursor = conn.execute("SELECT * FROM query_analysis")
            data = [dict(row) for row in cursor.fetchall()]

            if format == "json":
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)

            return output_file


# Example usage function
def example_usage():
    """Example of how to use the RAGLogger"""
    logger = RAGLogger()

    # Example configuration from your interface
    config = {
        'llm_provider': 'groq',
        'model': 'llama-3.1-70b-versatile',
        'temp': 0.1,
        'embedding_provider': 'huggingface',
        'embedding_model': 'all-MiniLM-L6-v2',
        'embedding_device': 'cpu',
        'vector_db': 'faiss',
        'chunk_size': 1000,
        'overlap': 200,
        'top_k': 4,
        'enhanced': True
    }

    # Log configuration
    config_id = logger.log_configuration(config)

    # Create session
    session_id = logger.create_session(config_id, "Test Session")

    # Example chunks from your interface
    chunks = [
        {
            'content': 'Example chunk content...',
            'similarity_score': 0.85,
            'rank': 1,
            'relevance_percentage': 85.0,
            'metadata': {'page': 1, 'source': 'test.pdf'}
        }
    ]

    # Log query and response
    query_id = logger.log_query_response(
        session_id=session_id,
        config_id=config_id,
        question="What is the main topic?",
        answer="The main topic is...",
        retrieved_chunks=chunks,
        response_time=1.5
    )

    print(f"Logged query: {query_id}")

    # Get comparison data
    comparison = logger.get_config_comparison()
    print("Configuration comparison:", comparison)


if __name__ == "__main__":
    example_usage()