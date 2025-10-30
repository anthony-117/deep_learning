from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import tempfile
import shutil
import asyncio
import json
from pathlib import Path
import logging

# Import your existing RAG components
from src.embedding import EmbeddingModel
from src.llm import LLMModel
from src.vectordb import VectorStore
from src.graph import RAGGraph
from src.ingest import ingest_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3002"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store initialized components
rag_graph: Optional[RAGGraph] = None
vector_store: Optional[VectorStore] = None
embedding_model: Optional[EmbeddingModel] = None
llm_model: Optional[LLMModel] = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize RAG components on startup."""
    global rag_graph, vector_store, embedding_model, llm_model
    
    try:
        logger.info("Initializing RAG system...")
        
        # Initialize components
        embedding_model = EmbeddingModel()
        llm_model = LLMModel()
        vector_store = VectorStore(embedding=embedding_model.get_embedding())
        
        # Try to initialize RAG graph
        try:
            rag_graph = RAGGraph(
                vector_store=vector_store,
                llm=llm_model,
                max_rewrites=2,
                relevance_threshold=0.5
            )
            logger.info("RAG system initialized successfully")
        except RuntimeError as e:
            logger.warning(f"Vector store not initialized: {e}")
            logger.info("RAG system ready for document ingestion")
            
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")

@app.get("/")
async def root():
    return {"message": "RAG API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rag_ready": rag_graph is not None,
        "vector_store_ready": vector_store is not None,
        "embedding_ready": embedding_model is not None,
        "llm_ready": llm_model is not None
    }

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process documents."""
    global rag_graph, vector_store, embedding_model, llm_model
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save uploaded files
            saved_files = []
            for file in files:
                if not file.filename:
                    continue
                    
                file_path = temp_path / file.filename
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                saved_files.append(str(file_path))
            
            if not saved_files:
                raise HTTPException(status_code=400, detail="No valid files uploaded")
            
            # Process documents through ingestion pipeline
            logger.info(f"Processing {len(saved_files)} files...")
            
            # Initialize vector store if not already done
            if vector_store is None:
                if embedding_model is None:
                    embedding_model = EmbeddingModel()
                vector_store = VectorStore(embedding=embedding_model.get_embedding())
            
            # Run ingestion pipeline
            processed_vector_store = ingest_pipeline(
                directory_path=str(temp_path),
                recursive=False,
                batch_size=50,
                embedding=embedding_model.get_embedding(),
                drop_old=False,
            )
            
            # Update global vector store
            vector_store = processed_vector_store
            
            # Initialize RAG graph with the new vector store
            if llm_model is None:
                llm_model = LLMModel()
            
            rag_graph = RAGGraph(
                vector_store=vector_store,
                llm=llm_model,
                max_rewrites=2,
                relevance_threshold=0.5
            )
            
            logger.info("Documents processed successfully")
            
            return {
                "message": "Documents uploaded and processed successfully",
                "files_processed": len(saved_files),
                "filenames": [Path(f).name for f in saved_files]
            }
            
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")

@app.post("/chat")
async def chat(question: str, chat_history: Optional[List[dict]] = None):
    """Process a chat question through the RAG system."""
    global rag_graph
    
    if not rag_graph:
        raise HTTPException(status_code=400, detail="RAG system not initialized. Please upload documents first.")
    
    try:
        # Convert chat history to LangChain messages
        from langchain_core.messages import HumanMessage, AIMessage
        
        history = []
        if chat_history:
            for msg in chat_history:
                if msg.get("role") == "user":
                    history.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    history.append(AIMessage(content=msg.get("content", "")))
        
        # Process question through RAG graph
        result = rag_graph.invoke(question, chat_history=history)
        
        return {
            "answer": result.get("answer", "I couldn't find a relevant answer."),
            "documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in result.get("documents", [])
            ],
            "steps": result.get("steps", []),
            "rewrites": result.get("rewrites", 0),
            "grounded": result.get("grounded", False)
        }
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat with step visualization."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            question = message_data.get("question")
            chat_history = message_data.get("chat_history", [])
            
            if not question:
                await manager.send_personal_message(
                    json.dumps({"error": "No question provided"}), 
                    websocket
                )
                continue
            
            if not rag_graph:
                await manager.send_personal_message(
                    json.dumps({"error": "RAG system not initialized"}), 
                    websocket
                )
                continue
            
            try:
                # Convert chat history
                from langchain_core.messages import HumanMessage, AIMessage
                
                history = []
                if chat_history:
                    for msg in chat_history:
                        if msg.get("role") == "user":
                            history.append(HumanMessage(content=msg.get("content", "")))
                        elif msg.get("role") == "assistant":
                            history.append(AIMessage(content=msg.get("content", "")))
                
                # Stream the RAG graph execution
                initial_state = {
                    "question": question,
                    "generation": None,
                    "documents": [],
                    "steps": [],
                    "rewrite_count": 0,
                    "relevance_scores": [],
                    "answer_grounded": False,
                    "chat_history": history
                }
                
                # Send initial state
                await manager.send_personal_message(
                    json.dumps({
                        "type": "step",
                        "step": "Starting RAG process...",
                        "state": initial_state
                    }), 
                    websocket
                )
                
                # Stream through the graph
                for output in rag_graph.app.stream(initial_state):
                    step_data = {
                        "type": "step_update",
                        "output": output,
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    await manager.send_personal_message(
                        json.dumps(step_data), 
                        websocket
                    )
                
                # Send final result
                final_result = rag_graph.invoke(question, chat_history=history)
                await manager.send_personal_message(
                    json.dumps({
                        "type": "final_result",
                        "result": {
                            "answer": final_result.get("answer", "I couldn't find a relevant answer."),
                            "documents": [
                                {
                                    "content": doc.page_content,
                                    "metadata": doc.metadata
                                } for doc in final_result.get("documents", [])
                            ],
                            "steps": final_result.get("steps", []),
                            "rewrites": final_result.get("rewrites", 0),
                            "grounded": final_result.get("grounded", False)
                        }
                    }), 
                    websocket
                )
                
            except Exception as e:
                logger.error(f"Error in WebSocket chat: {e}")
                await manager.send_personal_message(
                    json.dumps({"error": f"Error processing question: {str(e)}"}), 
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
