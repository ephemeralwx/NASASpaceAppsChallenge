#!/bin/bash

echo "🚀 nasa space biology research explorer setup"
echo "=============================================="
echo ""

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "found python $python_version"
echo ""

python3 -m venv env
echo "✓ virtual environment created"
echo ""

source env/bin/activate
echo "✓ activated"
echo ""

pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

echo "installing dependencies (this may take a few minutes)..."
echo ""

# torch 2.2.0 specifically - newer versions break pyg compatibility
echo "installing pytorch for mac..."
pip install torch==2.2.0 torchvision torchaudio

echo "installing pytorch geometric..."
pip install torch-geometric
# pyg extras need explicit cpu wheel url since mac has no cuda
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

echo "installing other packages..."
pip install streamlit transformers sentence-transformers pandas numpy scikit-learn \
    umap-learn plotly beautifulsoup4 requests lxml faiss-cpu rank-bm25 \
    networkx pyvis spacy scispacy aiohttp tqdm

echo ""
echo "downloading spacy biomedical model..."
# scispacy model hosted on s3, not pypi
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz

echo ""
echo "=============================================="
echo "✅ setup complete!"
echo ""
echo "to run the app:"
echo "  1. activate the virtual environment: source env/bin/activate"
echo "  2. run streamlit: streamlit run app.py"
echo ""
echo "the app will open in your browser at http://localhost:8501"
echo ""
echo "tips for m2 pro:"
echo "  - pytorch will use cpu (no cuda on mac)"
echo "  - first run will download models (~500mb)"
echo "  - embedding generation may take 5-10 min for 608 papers"
echo "  - use all-MiniLM-L6-v2 model for faster performance"
echo ""
echo "enjoy exploring nasa space biology research! 🚀"