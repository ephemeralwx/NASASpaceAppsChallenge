#!/bin/bash

echo "🚀 starting nasa space biology research explorer..."
echo ""

if [ ! -d "env" ]; then
    echo "❌ virtual environment not found!"
    echo "run ./setup.sh first"
    exit 1
fi

echo "activating virtual environment..."
source env/bin/activate

# using import check instead of pip list because it's faster and more reliable
if ! python -c "import streamlit" 2>/dev/null; then
    echo "❌ dependencies not installed!"
    echo "run ./setup.sh first"
    exit 1
fi

mkdir -p data

# cache can get corrupted with model changes, so manual clear option needed
if [ "$1" == "--clear-cache" ]; then
    echo "clearing cache..."
    rm -f data/*.npy data/*.pkl data/*.csv
    echo "✓ cache cleared"
fi

# test mode uses lighter models to avoid long startup times during dev
if [ "$1" == "--test" ]; then
    echo "running in test mode (faster model)..."
    export TEST_MODE=1
fi

echo ""
echo "launching streamlit app..."
echo "opening browser at http://localhost:8501"
echo ""
echo "press ctrl+c to stop"
echo ""

# headless mode prevents browser auto-opening which can be annoying in dev
streamlit run app.py --server.headless true

# cleanup happens automatically on script exit but being explicit
deactivate