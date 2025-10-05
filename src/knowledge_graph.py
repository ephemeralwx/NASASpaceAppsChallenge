import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx
from pyvis.network import Network
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import tempfile

class GNNModel(torch.nn.Module):
    def __init__(self, num_features, hidden_channels=64):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # outputting single value per node is a simplification - could be multi-dimensional
        self.conv3 = GCNConv(hidden_channels, 1)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # 0.3 dropout chosen empirically, no proper hyperparameter tuning
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x.squeeze()

class KnowledgeGraph:
    def __init__(self, df, embeddings):
        self.df = df
        self.embeddings = embeddings
        self.graph = None
        self.nx_graph = nx.Graph()
        self.pyg_data = None
        self.gnn_model = None
        self.communities = []
    
    def build_graph(self, similarity_threshold=0.7):
        print("building knowledge graph...")
        
        for idx, row in self.df.iterrows():
            self.nx_graph.add_node(
                idx,
                # truncating title for vis readability, keeping full version separately
                title=row['Title'][:50],
                year=row.get('year', 2020),
                full_title=row['Title']
            )
        
        print("computing semantic similarities...")
        sim_matrix = cosine_similarity(self.embeddings)
        
        edge_count = 0
        for i in range(len(sim_matrix)):
            for j in range(i+1, len(sim_matrix)):
                if sim_matrix[i][j] > similarity_threshold:
                    self.nx_graph.add_edge(i, j, weight=float(sim_matrix[i][j]))
                    edge_count += 1
        
        print(f"created graph with {self.nx_graph.number_of_nodes()} nodes and {edge_count} edges")
        
        self._detect_communities()
        self._build_pyg_data()
        self._train_gnn()
    
    def _detect_communities(self):
        # using dbscan instead of networkx community detection for consistency with embeddings
        clustering = DBSCAN(eps=0.3, min_samples=3)
        labels = clustering.fit_predict(self.embeddings)
        
        for idx, label in enumerate(labels):
            self.nx_graph.nodes[idx]['cluster'] = int(label)
        
        self.communities = list(set(labels))
        print(f"detected {len(self.communities)} communities")
    
    def _build_pyg_data(self):
        x = torch.tensor(self.embeddings, dtype=torch.float)
        
        edge_list = list(self.nx_graph.edges())
        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            # fallback to self-loops when no edges exist to avoid gnn errors
            edge_index = torch.tensor([[i, i] for i in range(len(self.df))], dtype=torch.long).t().contiguous()
        
        self.pyg_data = Data(x=x, edge_index=edge_index)
        print(f"pyg data created: {self.pyg_data}")
    
    def _train_gnn(self, epochs=50):
        print("training gnn for node importance...")
        
        num_features = self.embeddings.shape[1]
        self.gnn_model = GNNModel(num_features, hidden_channels=64)
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01)
        
        self.gnn_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            out = self.gnn_model(self.pyg_data.x, self.pyg_data.edge_index)
            
            # hacky unsupervised loss - using node degree as proxy for importance
            degrees = torch.tensor([self.nx_graph.degree(i) for i in range(len(self.df))], dtype=torch.float)
            # normalizing to prevent gradient explosion with high degree nodes
            degrees = (degrees - degrees.mean()) / (degrees.std() + 1e-8)
            
            loss = F.mse_loss(out, degrees)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"epoch {epoch+1}/{epochs}, loss: {loss.item():.4f}")
        
        self.gnn_model.eval()
        print("gnn training complete!")
    
    def compute_node_importance(self):
        if self.gnn_model is None:
            return []
        
        with torch.no_grad():
            importance_scores = self.gnn_model(self.pyg_data.x, self.pyg_data.edge_index)
            importance_scores = importance_scores.numpy()
        
        ranked = [(self.df.iloc[i]['Title'], float(importance_scores[i])) 
                 for i in range(len(importance_scores))]
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def create_interactive_graph(self, max_nodes=100):
        # performance compromise - large graphs are unusable in browser
        if len(self.nx_graph.nodes()) > max_nodes:
            top_nodes = sorted(self.nx_graph.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
            subgraph = self.nx_graph.subgraph([n[0] for n in top_nodes])
        else:
            subgraph = self.nx_graph
        
        net = Network(height='600px', width='100%', bgcolor='#1e1e1e', font_color='white')
        # physics params tuned for paper similarity graphs, not general purpose
        net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=100)
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b', '#fa709a']
        for node in subgraph.nodes():
            cluster = subgraph.nodes[node].get('cluster', 0)
            color = colors[cluster % len(colors)]
            title = subgraph.nodes[node]['full_title']
            
            net.add_node(
                node,
                label=subgraph.nodes[node]['title'],
                title=title,
                color=color,
                # arbitrary scaling factor for node size visualization
                size=10 + subgraph.degree(node) * 2
            )
        
        for edge in subgraph.edges():
            weight = subgraph.edges[edge].get('weight', 0.5)
            net.add_edge(edge[0], edge[1], value=weight, color='rgba(100,100,100,0.3)')
        
        # using tempfile to avoid html file management issues
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.html') as f:
            net.save_graph(f.name)
            f.seek(0)
            with open(f.name, 'r') as f2:
                html = f2.read()
        
        return html
    
    def num_nodes(self):
        return self.nx_graph.number_of_nodes()
    
    def num_edges(self):
        return self.nx_graph.number_of_edges()
    
    def num_communities(self):
        return len(self.communities)