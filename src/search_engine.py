import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import faiss

class HybridSearchEngine:
    def __init__(self, df, embeddings, embedding_engine):
        self.df = df
        self.embeddings = embeddings
        self.embedding_engine = embedding_engine
        
        print("building bm25 index...")
        corpus = []
        for idx, row in df.iterrows():
            doc = f"{row['Title']} {row.get('abstract', '')}"
            corpus.append(doc.lower().split())
        self.bm25 = BM25Okapi(corpus)
        
        print("building faiss index...")
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(normalized_embeddings.astype('float32'))
        
        print("search engine ready!")
    
    def hybrid_search(self, query, k=20, alpha=0.5, query_expansion=True):
        if query_expansion:
            expanded_query = self.embedding_engine.expand_query(query)
            print(f"expanded query: {expanded_query}")
        else:
            expanded_query = query
        
        sparse_scores = self._bm25_search(expanded_query, k=k*2)
        
        dense_scores = self._dense_search(expanded_query, k=k*2)
        
        fused_scores = self._fusion(sparse_scores, dense_scores, alpha=alpha)
        
        top_k_indices = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for idx, score in top_k_indices:
            results.append({
                'title': self.df.iloc[idx]['Title'],
                'abstract': self.df.iloc[idx].get('abstract', '')[:300],
                'link': self.df.iloc[idx].get('Link', ''),
                'score': float(score),
                'year': self.df.iloc[idx].get('year', 0)
            })
        
        return results
    
    def _bm25_search(self, query, k=40):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return {i: scores[i] for i in range(len(scores))}
    
    def _dense_search(self, query, k=40):
        query_embedding = self.embedding_engine.encode_query(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        distances, indices = self.index.search(query_embedding, k)
        
        scores_dict = {}
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            scores_dict[int(idx)] = float(dist)
        
        return scores_dict
    
    def _fusion(self, sparse_scores, dense_scores, alpha=0.5):
        all_indices = set(sparse_scores.keys()) | set(dense_scores.keys())
        
        fused = {}
        for idx in all_indices:
            sparse_score = sparse_scores.get(idx, 0)
            dense_score = dense_scores.get(idx, 0)
            
            fused[idx] = alpha * dense_score + (1 - alpha) * sparse_score
        
        return fused
    
    def semantic_search(self, query, k=20):
        return self.hybrid_search(query, k=k, alpha=1.0, query_expansion=False)
    
    def keyword_search(self, query, k=20):
        return self.hybrid_search(query, k=k, alpha=0.0, query_expansion=False)