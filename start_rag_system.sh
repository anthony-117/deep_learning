#!/bin/bash

# RAG System Startup Script
# This script starts both the FastAPI backend and Next.js frontend

echo "🚀 Starting RAG System..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Check if uv is available
if command -v uv >/dev/null 2>&1; then
    echo "📦 Using uv package manager..."
else
    echo "📦 Activating Python virtual environment..."
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "⚠️  No virtual environment found and uv not available"
        echo "Please either install uv or create a virtual environment"
        exit 1
    fi
fi

# Install FastAPI dependencies
echo "📥 Installing FastAPI dependencies..."
if command -v uv >/dev/null 2>&1; then
    echo "Using uv package manager..."
    uv add fastapi uvicorn python-multipart websockets
else
    echo "uv not found, trying pip in virtual environment..."
    if [ -d "venv" ]; then
        venv/bin/pip install fastapi uvicorn python-multipart websockets
    else
        echo "Please install uv or set up a virtual environment"
        exit 1
    fi
fi

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo "❌ Frontend directory not found. Please ensure the frontend is set up."
    exit 1
fi

# Install frontend dependencies
echo "📥 Installing frontend dependencies..."
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
fi
cd ..

# Function to cleanup background processes
cleanup() {
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start backend server
echo "🔧 Starting FastAPI backend server on http://localhost:8000..."
if command -v uv >/dev/null 2>&1; then
    uv run python api_server.py &
else
    if [ -d "venv" ]; then
        venv/bin/python api_server.py &
    else
        python api_server.py &
    fi
fi
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend server
echo "🎨 Starting Next.js frontend server on http://localhost:3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ RAG System is now running!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
