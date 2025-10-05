"""
utility functions for nasa space biology explorer
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
import hashlib
import time
from functools import wraps

def cache_to_disk(cache_dir="data", filename=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            Path(cache_dir).mkdir(exist_ok=True)
            
            if filename is None:
                cache_file = f"{cache_dir}/{func.__name__}_cache.pkl"
            else:
                cache_file = f"{cache_dir}/{filename}"
            
            if Path(cache_file).exists():
                print(f"loading cached result from {cache_file}")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            print(f"computing {func.__name__}...")
            result = func(*args, **kwargs)
            
            print(f"saving to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            return result
        return wrapper
    return decorator

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # avoid division by zero in degenerate cases
    return embeddings / (norms + 1e-8)

def compute_cosine_similarity(emb1, emb2):
    emb1_norm = normalize_embeddings(emb1.reshape(1, -1))
    emb2_norm = normalize_embeddings(emb2.reshape(1, -1))
    return np.dot(emb1_norm, emb2_norm.T)[0, 0]

def batch_iterator(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def extract_keywords(text, n=10):
    # quick and dirty frequency-based approach, good enough for exploration
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that'}
    
    words = text.lower().split()
    # arbitrary min length to filter noise
    words = [w for w in words if w not in stopwords and len(w) > 3]
    
    from collections import Counter
    word_counts = Counter(words)
    
    return [word for word, count in word_counts.most_common(n)]

def format_large_number(num):
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(num)

def truncate_text(text, max_length=100, suffix="..."):
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

def create_year_bins(df, year_col='year', bins=5):
    min_year = df[year_col].min()
    max_year = df[year_col].max()
    
    bin_edges = np.linspace(min_year, max_year, bins + 1)
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" 
                 for i in range(len(bin_edges)-1)]
    
    df['year_bin'] = pd.cut(df[year_col], bins=bin_edges, labels=bin_labels)
    return df

def deduplicate_papers(df, title_col='Title', threshold=0.9):
    # sequencematcher is slow but good enough for moderate datasets
    from difflib import SequenceMatcher
    
    keep_indices = []
    seen_titles = []
    
    for idx, title in enumerate(df[title_col]):
        is_duplicate = False
        for seen_title in seen_titles:
            similarity = SequenceMatcher(None, title.lower(), seen_title.lower()).ratio()
            if similarity > threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            keep_indices.append(idx)
            seen_titles.append(title)
    
    print(f"removed {len(df) - len(keep_indices)} duplicates")
    return df.iloc[keep_indices].reset_index(drop=True)

def extract_year_from_text(text):
    import re
    # assumes years 2000-2029, hardcoded for space research papers
    match = re.search(r'20[0-2][0-9]', text)
    if match:
        return int(match.group())
    return None

def compute_diversity_score(embeddings):
    from sklearn.metrics.pairwise import cosine_similarity
    
    sim_matrix = cosine_similarity(embeddings)
    
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    avg_similarity = sim_matrix[mask].mean()
    
    diversity = 1 - avg_similarity
    return diversity

def identify_outliers(embeddings, contamination=0.1):
    # isolation forest works well for high-dim embeddings
    from sklearn.ensemble import IsolationForest
    
    clf = IsolationForest(contamination=contamination, random_state=42)
    outlier_labels = clf.fit_predict(embeddings)
    
    outlier_indices = np.where(outlier_labels == -1)[0]
    return outlier_indices

def create_word_cloud_data(texts, max_words=50):
    from collections import Counter
    import re
    
    combined = ' '.join(texts).lower()
    
    # min 4 chars to avoid noise like "the", "and"
    words = re.findall(r'\b[a-z]{4,}\b', combined)
    
    stopwords = {'that', 'this', 'with', 'from', 'have', 'were', 'been', 
                'their', 'which', 'these', 'about', 'would', 'there', 'other'}
    
    words = [w for w in words if w not in stopwords]
    
    word_counts = Counter(words).most_common(max_words)
    return dict(word_counts)

def export_to_csv(df, filepath, columns=None):
    if columns:
        df = df[columns]
    df.to_csv(filepath, index=False)
    print(f"exported to {filepath}")

def export_embeddings(embeddings, df, filepath):
    np.savez(
        filepath,
        embeddings=embeddings,
        titles=df['Title'].values,
        # fallback year if column missing
        years=df.get('year', pd.Series([2020]*len(df))).values
    )
    print(f"exported embeddings to {filepath}")

def load_embeddings(filepath):
    data = np.load(filepath)
    return {
        'embeddings': data['embeddings'],
        'titles': data['titles'],
        'years': data['years']
    }

class ProgressTracker:
    def __init__(self, total, desc="Processing"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = time.time()
    
    def update(self, n=1):
        self.current += n
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / rate if rate > 0 else 0
        
        percent = (self.current / self.total) * 100
        print(f"\r{self.desc}: {self.current}/{self.total} "
              f"({percent:.1f}%) - {rate:.1f} it/s - "
              f"ETA: {remaining:.0f}s", end='')
        
        if self.current >= self.total:
            print()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if self.current < self.total:
            print()

def validate_dataframe(df, required_columns):
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    return True

def safe_divide(a, b, default=0):
    try:
        return a / b if b != 0 else default
    except:
        return default

def get_memory_usage():
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    return mem_mb

def print_system_info():
    import torch
    import platform
    
    print("system information:")
    print(f"  python: {platform.python_version()}")
    print(f"  platform: {platform.platform()}")
    print(f"  pytorch: {torch.__version__}")
    print(f"  cuda available: {torch.cuda.is_available()}")
    # mps backend is newer and might not exist
    if hasattr(torch.backends, 'mps'):
        print(f"  mps available: {torch.backends.mps.is_available()}")
    print(f"  memory usage: {get_memory_usage():.1f} MB")