#!/bin/bash
# Alzheimer Disease Prediction System - Linux/Mac Startup Script

echo ""
echo "================================"
echo "Alzheimer Disease Prediction"
echo "IoMT & Deep Learning System"
echo "================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Initialize database if needed
if [ ! -f "alzheimer_db.db" ]; then
    echo "Initializing database..."
    python setup.py
fi

# Start the application
echo ""
echo "================================"
echo "Starting application..."
echo "================================"
echo ""
echo "Access the application at: http://localhost:5000"
echo ""
echo "Admin Login:"
echo "  Username: admin"
echo "  Password: admin123"
echo ""
echo "Demo User Login:"
echo "  Username: demo"
echo "  Password: demo123"
echo ""

python run.py
