# RAG Chatbot Frontend

A modern, responsive web interface for the RAG (Retrieval-Augmented Generation) chatbot built with Next.js, TypeScript, and Tailwind CSS.

## Features

- **📁 Upload Page**: Drag-and-drop file upload with progress tracking
- **💬 Chat Interface**: Real-time chat with step-by-step process visualization
- **📊 Process Visualization**: See embedding, retrieval, generation, and other RAG steps in real-time
- **📄 Document Management**: View and manage uploaded documents
- **⚙️ Settings**: Configure system parameters and view status
- **🎨 Modern UI**: Clean, responsive design with smooth animations

## Tech Stack

- **Frontend**: Next.js 14, React 18, TypeScript
- **Styling**: Tailwind CSS, Lucide React icons
- **Backend**: FastAPI (Python)
- **Real-time**: WebSocket connections
- **File Upload**: React Dropzone

## Getting Started

### Prerequisites

- Node.js 18+ 
- Python 3.8+
- Your existing RAG system running

### Installation

1. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Install backend dependencies** (if not already installed):
   ```bash
   pip install fastapi uvicorn python-multipart websockets
   ```

3. **Start the backend server**:
   ```bash
   python api_server.py
   ```
   The API will be available at `http://localhost:8000`

4. **Start the frontend development server**:
   ```bash
   cd frontend
   npm run dev
   ```
   The frontend will be available at `http://localhost:3000`

### Usage

1. **Upload Documents**: Go to the upload page and drag & drop your documents (PDF, DOC, DOCX, TXT, MD)

2. **Start Chatting**: Navigate to the chat page and ask questions about your documents

3. **View Process Steps**: Enable "Show Steps" to see real-time visualization of:
   - Query analysis
   - Document retrieval
   - Relevance grading
   - Answer generation
   - Hallucination checking
   - Query rewriting (if needed)

4. **Manage Documents**: View uploaded documents and their processing status

5. **Configure Settings**: Adjust system parameters and view system health

## Project Structure

```
frontend/
├── app/
│   ├── chat/page.tsx          # Chat interface with real-time steps
│   ├── documents/page.tsx     # Document management
│   ├── settings/page.tsx      # System configuration
│   ├── layout.tsx             # Root layout with navigation
│   ├── page.tsx               # Upload page
│   └── globals.css            # Global styles
├── components/
│   └── Navigation.tsx         # Main navigation component
├── package.json
├── tailwind.config.js
└── tsconfig.json
```

## API Endpoints

The FastAPI backend provides these endpoints:

- `GET /health` - System health check
- `POST /upload` - Upload and process documents
- `POST /chat` - Send chat messages
- `WS /ws` - WebSocket for real-time chat with step visualization

## Environment Variables

Make sure you have these environment variables set:

- `GROQ_API_KEY` - Your GROQ API key
- `MILVUS_HOST` - Milvus vector database host (default: localhost)
- `MILVUS_PORT` - Milvus port (default: 19530)
- `VECTOR_DB_COLLECTION` - Collection name (default: docling_demo)

## Features in Detail

### Real-time Process Visualization

The chat interface shows each step of the RAG process in real-time:

1. **🔍 Analyzing Query** - Understanding the user's question
2. **📚 Retrieving Documents** - Finding relevant documents from vector store
3. **📊 Grading Document Relevance** - Evaluating document relevance
4. **🤖 Generating Answer** - Creating the response using LLM
5. **🔍 Checking for Hallucinations** - Verifying answer accuracy
6. **✏️ Rewriting Query** - Improving queries when needed

### Upload Process

- Drag & drop multiple files
- Support for PDF, DOC, DOCX, TXT, MD formats
- Real-time upload progress
- Document processing status
- Batch processing with progress tracking

### Chat Features

- Real-time messaging
- Message history
- Source document citations
- Processing step details
- Query rewrite tracking
- Grounded answer verification

## Development

### Adding New Features

1. Create new pages in `app/` directory
2. Add components in `components/` directory
3. Update navigation in `components/Navigation.tsx`
4. Add API endpoints in `api_server.py`

### Styling

- Uses Tailwind CSS for styling
- Custom components defined in `globals.css`
- Responsive design with mobile-first approach
- Dark/light theme support ready

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**: Ensure the backend is running on port 8000
2. **Upload Fails**: Check file format and size limits
3. **Chat Not Working**: Verify RAG system is initialized with documents
4. **Styling Issues**: Ensure Tailwind CSS is properly configured

### Debug Mode

Enable debug logging by setting `NODE_ENV=development` in your environment.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of your RAG system implementation.
