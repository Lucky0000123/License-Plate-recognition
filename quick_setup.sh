#!/bin/bash

# Quick Setup Script for License Plate Recognition System
# This script automates the entire setup process

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}→ $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main setup
print_header "License Plate Recognition - Quick Setup"

# Check prerequisites
print_info "Checking prerequisites..."

if ! command_exists python3; then
    print_error "Python 3 is not installed"
    exit 1
fi
print_success "Python 3 found: $(python3 --version)"

if ! command_exists node; then
    print_error "Node.js is not installed"
    exit 1
fi
print_success "Node.js found: $(node --version)"

if ! command_exists npm; then
    print_error "npm is not installed"
    exit 1
fi
print_success "npm found: $(npm --version)"

# Create virtual environment
print_header "Setting up Python Virtual Environment"

if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Install Python dependencies
print_header "Installing Python Dependencies"
print_info "This may take a few minutes..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
print_success "Python dependencies installed"

# Install frontend dependencies
print_header "Installing Frontend Dependencies"
cd frontend
if [ ! -d "node_modules" ]; then
    print_info "Installing npm packages..."
    npm install
    print_success "Frontend dependencies installed"
else
    print_warning "node_modules already exists, skipping..."
fi
cd ..

# Create necessary directories
print_header "Creating Directory Structure"
mkdir -p backend/saved_models
mkdir -p data/raw/images
mkdir -p data/raw/annotations
mkdir -p data/processed
mkdir -p demo_output
print_success "Directories created"

# Generate sample images
print_header "Generating Sample Images"
print_info "Creating test images..."
python3 create_sample_images.py
print_success "Sample images generated"

# Run validation
print_header "Validating Setup"
python3 validate_setup.py

# Summary
print_header "Setup Complete!"
echo ""
echo -e "${GREEN}Your License Plate Recognition system is ready!${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. ${YELLOW}Train models (optional):${NC}"
echo "   python backend/training/train_detector.py --use-sample-data --epochs 10"
echo "   python backend/training/train_ocr.py --use-sample-data --epochs 20"
echo ""
echo "2. ${YELLOW}Run demo:${NC}"
echo "   python demo.py"
echo ""
echo "3. ${YELLOW}Start the application:${NC}"
echo "   ./run_app.sh"
echo ""
echo "4. ${YELLOW}Or use Docker:${NC}"
echo "   docker-compose up --build"
echo ""
echo -e "${BLUE}Happy coding! 🚀${NC}"
echo ""

