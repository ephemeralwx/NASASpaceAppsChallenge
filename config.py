"""
configuration file for nasa space biology research explorer
modify these settings to customize behavior
"""

# data source

DATA_SOURCE = "https://raw.githubusercontent.com/jgalazka/SB_publications/main/SB_publication_PMC.csv"
CACHE_DIR = "data"
CACHE_ENABLED = True

# embedding models
DEFAULT_MODEL = "BioLinkBERT-base"
EMBEDDING_BATCH_SIZE = 8
MAX_SEQUENCE_LENGTH = 512

# knowledge graph
# balance between precision and recall for research papers
KG_SIMILARITY_THRESHOLD = 0.7
KG_MAX_NODES_DISPLAY = 100
# values tuned empirically for academic paper clustering behavior
DBSCAN_EPS = 0.3
DBSCAN_MIN_SAMPLES = 3

# gnn training
GNN_HIDDEN_CHANNELS = 64
GNN_DROPOUT = 0.3
GNN_LEARNING_RATE = 0.01
GNN_EPOCHS = 50

# search engine
SEARCH_DEFAULT_K = 20
# hybrid search weight - pure semantic vs keyword matching
SEARCH_ALPHA = 0.5
FAISS_USE_GPU = False
QUERY_EXPANSION_ENABLED = True

# claim extraction
# constrain output to avoid overwhelming users
MAX_CLAIMS_PER_PAPER = 3
MIN_CLAIM_LENGTH = 10
MAX_CLAIM_LENGTH = 50
CLAIM_BATCH_SIZE = 100

# visualization
# umap params chosen for biological literature clustering patterns
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = 'cosine'
UMAP_RANDOM_STATE = 42

KMEANS_N_CLUSTERS = 8
PLOTLY_HEIGHT = 700

# multi-agent system
ENABLE_ALL_AGENTS = True
AGENT_TIMEOUT = 30

# performance
USE_MULTIPROCESSING = False
N_JOBS = -1

# api rate limiting
# respect pmc api guidelines
PMC_RATE_LIMIT_DELAY = 0.5
# demo limit to avoid hitting api quotas
MAX_ABSTRACT_FETCHES = 50

# ui settings
DARK_MODE = True
GRADIENT_COLORS = ['#667eea', '#764ba2']
SHOW_DEBUG_INFO = False

# file paths
PUBLICATIONS_CACHE = f"{CACHE_DIR}/publications.csv"
EMBEDDINGS_CACHE = f"{CACHE_DIR}/embeddings.npy"
KG_CACHE = f"{CACHE_DIR}/knowledge_graph.pkl"

# model paths (will download if not present)
MODEL_PATHS = {
    'BioLinkBERT-base': 'michiyasunaga/BioLinkBERT-base',
    'SciBERT': 'allenai/scibert_scivocab_uncased',
    'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2'
}

# scispacy model
SCISPACY_MODEL = "en_core_sci_md"

# logging
LOG_LEVEL = "INFO"
LOG_FILE = "app.log"
LOG_TO_FILE = False