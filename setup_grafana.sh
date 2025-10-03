#!/bin/bash

# RAG Performance Dashboard - Grafana Setup Script
echo "🚀 Setting up Grafana dashboard for RAG performance monitoring..."

# Check if running on Ubuntu/Debian
if [[ -f /etc/debian_version ]]; then
    echo "📦 Installing Grafana on Debian/Ubuntu..."

    # Add Grafana repository
    sudo apt-get install -y software-properties-common wget
    sudo mkdir -p /etc/apt/keyrings/
    wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null
    echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list

    # Update and install
    sudo apt-get update
    sudo apt-get install -y grafana

    # Install SQLite plugin
    sudo grafana-cli plugins install frser-sqlite-datasource

    # Enable and start Grafana
    sudo systemctl daemon-reload
    sudo systemctl enable grafana-server
    sudo systemctl start grafana-server

    echo "✅ Grafana installed successfully!"
    echo "📊 Access your dashboard at: http://localhost:3000"
    echo "🔑 Default login: admin / admin"

elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "📦 Installing Grafana on macOS..."

    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi

    # Install Grafana
    brew update
    brew install grafana

    # Install SQLite plugin
    grafana-cli plugins install frser-sqlite-datasource

    # Start Grafana
    brew services start grafana

    echo "✅ Grafana installed successfully!"
    echo "📊 Access your dashboard at: http://localhost:3000"
    echo "🔑 Default login: admin / admin"

else
    echo "🐳 Using Docker installation (works on any OS)..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not found. Please install Docker first:"
        echo "   https://docs.docker.com/get-docker/"
        exit 1
    fi

    # Create Grafana directory
    mkdir -p grafana-storage

    # Run Grafana in Docker
    docker run -d \
        --name grafana-rag \
        -p 3000:3000 \
        -v "$(pwd)/grafana-storage:/var/lib/grafana" \
        -v "$(pwd):/data" \
        -e "GF_INSTALL_PLUGINS=frser-sqlite-datasource" \
        grafana/grafana:latest

    echo "✅ Grafana started in Docker!"
    echo "📊 Access your dashboard at: http://localhost:3000"
    echo "🔑 Default login: admin / admin"
fi

echo ""
echo "🔧 Next steps:"
echo "1. Open http://localhost:3000 in your browser"
echo "2. Login with admin/admin (you'll be asked to change password)"
echo "3. Run: python setup_grafana_datasource.py"
echo "4. Import the dashboard configuration"
echo ""
echo "💡 Your RAG database will be automatically connected!"