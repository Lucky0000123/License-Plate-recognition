#!/bin/bash

# License Plate Recognition System - Easy Startup Script
# This script starts both backend and frontend servers

echo "=========================================="
echo "License Plate Recognition System"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "Checking prerequisites..."

if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

if ! command_exists node; then
    echo -e "${RED}Error: Node.js is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites check passed${NC}"
echo ""

# Setup backend
echo "Setting up backend..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing backend dependencies..."
pip install -r requirements.txt -q

# Setup frontend
echo ""
echo "Setting up frontend..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
else
    echo "Frontend dependencies already installed"
fi

cd ..

# Start backend in background
echo ""
echo -e "${GREEN}Starting backend server...${NC}"
cd backend
python app.py &
BACKEND_PID=$!
cd ..

echo "Backend PID: $BACKEND_PID"
echo "Backend running on http://localhost:5000"

# Wait a bit for backend to start
sleep 3

# Start frontend
echo ""
echo -e "${GREEN}Starting frontend server...${NC}"
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "=========================================="
echo -e "${GREEN}Application started successfully!${NC}"
echo "=========================================="
echo ""
echo "Backend:  http://localhost:5000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Trap Ctrl+C and cleanup
trap cleanup INT

cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "Servers stopped"
    exit 0
}

# Wait for both processes
wait
