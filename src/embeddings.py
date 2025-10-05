import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

class EmbeddingEngine:
    def __init__(self, model_name="BioLinkBERT-base"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"using device: {self.device}")
        
        # two different apis because sentence-transformers is simpler but transformers gives more control
        if model_name == "BioLinkBERT-base":
            self.tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base")
            self.model = AutoModel.from_pretrained("michiyasunaga/BioLinkBERT-base")
            self.use_sentence_transformer = False
        elif model_name == "SciBERT":
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
            self.model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
            self.use_sentence_transformer = False
        else:  
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.use_sentence_transformer = True
        
        if not self.use_sentence_transformer:
            self.model = self.model.to(self.device)
            self.model.eval()
    
    def generate_embeddings(self, df, batch_size=8):
        texts = []
        
        for idx, row in df.iterrows():
            # abstract might be missing in some datasets, fallback to empty string
            text = f"{row['Title']} {row.get('abstract', '')}"
            # 512 is arbitrary but bert models usually cap at 512 tokens anyway
            texts.append(text[:512])  
        
        embeddings = []
        
        if self.use_sentence_transformer:
            print(f"generating embeddings with {self.model_name}...")
            embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=batch_size)
        else:
            print(f"generating embeddings with {self.model_name}...")
            for i in tqdm(range(0, len(texts), batch_size)):
                batch = texts[i:i+batch_size]
                batch_embeddings = self._encode_batch(batch)
                embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings)
        print(f"generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def _encode_batch(self, texts):
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            # using cls token instead of mean pooling - simpler and often works just as well
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings
    
    def encode_query(self, query):
        if self.use_sentence_transformer:
            return self.model.encode([query])[0]
        else:
            return self._encode_batch([query])[0]
    
    def expand_query(self, query, top_k=3):
        # hacky hardcoded expansion - should use wordnet or domain ontologies in production
        expansions = {
            'microgravity': ['weightlessness', 'zero gravity', 'reduced gravity'],
            'radiation': ['cosmic rays', 'ionizing radiation', 'space radiation'],
            'bone': ['skeletal', 'osseous', 'bone density'],
            'muscle': ['muscular', 'skeletal muscle', 'myofiber'],
            'plant': ['vegetation', 'flora', 'botanical'],
            'cell': ['cellular', 'cells', 'cytology'],
            'mars': ['martian', 'red planet'],
            'space': ['spaceflight', 'orbital', 'extraterrestrial']
        }
        
        expanded_terms = [query]
        words = query.lower().split()
        
        for word in words:
            if word in expansions:
                # limiting to top_k to avoid query explosion
                expanded_terms.extend(expansions[word][:top_k])
        
        return ' '.join(expanded_terms)