import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from umap import UMAP
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class Visualizer:
    def __init__(self, df, embeddings=None):
        self.df = df
        self.embeddings = embeddings
        self.umap_coords = None
        self.clusters = None
    
    def create_3d_landscape(self, color_by='cluster', n_neighbors=15):
        # if we don't have embeddings, can't create the landscape
        if self.embeddings is None:
            return self._create_placeholder_viz()
        
        # compute umap coordinates if not already done
        if self.umap_coords is None:
            print("computing umap projection...")
            umap_model = UMAP(
                n_components=3,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric='cosine',
                random_state=42
            )
            self.umap_coords = umap_model.fit_transform(self.embeddings)
            
            # arbitrary choice of 8 clusters - should be configurable but works for most datasets
            kmeans = KMeans(n_clusters=8, random_state=42)
            self.clusters = kmeans.fit_predict(self.embeddings)
        
        # prepare data for plotting
        plot_df = pd.DataFrame({
            'x': self.umap_coords[:, 0],
            'y': self.umap_coords[:, 1],
            'z': self.umap_coords[:, 2],
            'title': self.df['Title'].str[:50],
            'year': self.df.get('year', 2020),
            'cluster': self.clusters
        })
        
        # determine color scale based on user choice
        if color_by == 'cluster':
            color = plot_df['cluster']
            colorscale = 'rainbow'
        elif color_by == 'year':
            color = plot_df['year']
            colorscale = 'viridis'
        else:
            from sklearn.neighbors import KernelDensity
            # bandwidth=0.5 is a heuristic that works well for normalized umap coords
            kde = KernelDensity(bandwidth=0.5)
            kde.fit(self.umap_coords)
            density = np.exp(kde.score_samples(self.umap_coords))
            color = density
            colorscale = 'hot'
        
        # create 3d scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=plot_df['x'],
            y=plot_df['y'],
            z=plot_df['z'],
            mode='markers',
            marker=dict(
                size=5,
                color=color,
                colorscale=colorscale,
                showscale=True,
                opacity=0.8,
                line=dict(width=0.5, color='white')
            ),
            text=plot_df['title'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Year: %{customdata[0]}<br>' +
                         'Cluster: %{customdata[1]}<extra></extra>',
            customdata=plot_df[['year', 'cluster']].values
        )])
        
        fig.update_layout(
            title='3D Research Landscape - UMAP Projection',
            scene=dict(
                xaxis_title='UMAP-1',
                yaxis_title='UMAP-2',
                zaxis_title='UMAP-3',
                bgcolor='#1e1e1e'
            ),
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='white'),
            height=700
        )
        
        return fig
    
    def identify_mountains(self):
        # can't identify mountains without umap coords or clusters
        if self.umap_coords is None or self.clusters is None:
            return ["Compute 3D landscape first"]
        
        # count papers per cluster
        cluster_counts = pd.Series(self.clusters).value_counts()
        
        mountains = []
        for cluster_id in cluster_counts.head(5).index:
            cluster_papers = self.df[self.clusters == cluster_id]
            
            # hacky topic extraction - should use proper topic modeling but this works for demo
            sample_titles = ' '.join(cluster_papers['Title'].head(5).tolist())
            
            words = sample_titles.lower().split()
            word_freq = pd.Series(words).value_counts()
            # filter out short words since they're usually stop words or uninformative
            common_words = [w for w in word_freq.head(3).index if len(w) > 4]
            
            topic = ' + '.join(common_words) if common_words else f'Cluster {cluster_id}'
            mountains.append(f"{topic} ({cluster_counts[cluster_id]} papers)")
        
        return mountains
    
    def identify_valleys(self):
        # can't identify valleys without umap coords or clusters
        if self.umap_coords is None or self.clusters is None:
            return ["Compute 3D landscape first"]
        
        cluster_counts = pd.Series(self.clusters).value_counts()
        # threshold of 10 is arbitrary but works well to identify underexplored areas
        small_clusters = cluster_counts[cluster_counts < 10]
        
        valleys = []
        for cluster_id in small_clusters.head(5).index:
            cluster_papers = self.df[self.clusters == cluster_id]
            
            sample_titles = ' '.join(cluster_papers['Title'].head(3).tolist())
            words = sample_titles.lower().split()
            word_freq = pd.Series(words).value_counts()
            common_words = [w for w in word_freq.head(2).index if len(w) > 4]
            
            topic = ' + '.join(common_words) if common_words else f'Cluster {cluster_id}'
            valleys.append(f"{topic} (only {cluster_counts[cluster_id]} papers)")
        
        return valleys
    
    def create_topic_year_heatmap(self):
        # if we don't have clusters, fallback to simple keyword-based heatmap
        if self.clusters is None:
            return self._create_simple_year_heatmap()
        
        df_with_cluster = self.df.copy()
        df_with_cluster['cluster'] = self.clusters
        
        # group by year and cluster to create heatmap data
        heatmap_data = df_with_cluster.groupby(['year', 'cluster']).size().reset_index(name='count')
        
        # pivot the data for heatmap
        heatmap_pivot = heatmap_data.pivot(index='cluster', columns='year', values='count').fillna(0)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=[f'Topic {i}' for i in heatmap_pivot.index],
            colorscale='Blues',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Research Topic Evolution Over Time',
            xaxis_title='Year',
            yaxis_title='Research Topic Cluster',
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='white'),
            height=500
        )
        
        return fig
    
    def _create_simple_year_heatmap(self):
        # hardcoded keywords for space biology - should be configurable but these cover main areas
        keywords = ['microgravity', 'radiation', 'bone', 'muscle', 'plant', 'cell']
        
        heatmap_data = []
        for keyword in keywords:
            for year in range(2010, 2025):
                count = self.df[
                    (self.df['Title'].fillna('').str.lower().str.contains(keyword)) & 
                    (self.df['year'] == year)
                ].shape[0]
                heatmap_data.append({
                    'keyword': keyword,
                    'year': year,
                    'count': count
                })
        
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_pivot = heatmap_df.pivot(index='keyword', columns='year', values='count').fillna(0)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title='Keyword Frequency Over Time',
            xaxis_title='Year',
            yaxis_title='Keyword',
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='white'),
            height=400
        )
        
        return fig
    
    def create_collaboration_heatmap(self):
        # prefer real cluster collaboration if we have embeddings, else fallback to keyword co-occurrence
        if self.clusters is not None and self.embeddings is not None:
            return self._create_cluster_collaboration_heatmap()
        else:
            return self._create_keyword_cooccurrence_heatmap()
    
    def _create_cluster_collaboration_heatmap(self):
        print("computing cross-topic collaboration patterns...")
        
        unique_clusters = sorted(set(self.clusters))
        n_clusters = len(unique_clusters)
        
        # initialize collaboration matrix for clusters
        collab_matrix = np.zeros((n_clusters, n_clusters))
        
        # for each pair of clusters, compute average cross-cluster similarity
        for i, cluster_i in enumerate(unique_clusters):
            for j, cluster_j in enumerate(unique_clusters):
                if i == j:
                    # same cluster - set to 0 (we're interested in cross-cluster collaboration)
                    collab_matrix[i, j] = 0
                else:
                    embeddings_i = self.embeddings[self.clusters == cluster_i]
                    embeddings_j = self.embeddings[self.clusters == cluster_j]
                    
                    if len(embeddings_i) > 0 and len(embeddings_j) > 0:
                        cross_sim = cosine_similarity(embeddings_i, embeddings_j)
                        
                        # 0.7 threshold chosen empirically - captures strong semantic overlap without noise
                        high_similarity_count = np.sum(cross_sim > 0.7)
                        
                        max_possible = min(len(embeddings_i), len(embeddings_j))
                        if max_possible > 0:
                            collab_matrix[i, j] = high_similarity_count
        
        # make matrix symmetric
        collab_matrix = (collab_matrix + collab_matrix.T) / 2
        
        # generate topic labels based on cluster content
        topic_labels = []
        for cluster_id in unique_clusters:
            cluster_papers = self.df[self.clusters == cluster_id]
            
            # extract representative keywords for this cluster
            all_text = ' '.join(cluster_papers['Title'].fillna('').head(10).tolist()).lower()
            
            # predefined topics for space biology domain - could be learned but this is more reliable
            topic_keywords = {
                'Microgravity': ['microgravity', 'weightless', 'zero-g', 'gravity'],
                'Radiation': ['radiation', 'cosmic', 'ionizing'],
                'Bone/Skeletal': ['bone', 'skeletal', 'osteo'],
                'Muscle': ['muscle', 'muscular', 'myofiber'],
                'Plant Biology': ['plant', 'vegetation', 'botanical', 'growth'],
                'Immunology': ['immune', 'immunity', 'immunological'],
                'Cardiovascular': ['cardiovascular', 'heart', 'vascular', 'cardiac'],
                'Cell Biology': ['cell', 'cellular', 'cytology', 'molecular']
            }
            
            best_topic = f'Topic {cluster_id}'
            max_matches = 0
            
            for topic_name, keywords in topic_keywords.items():
                matches = sum([all_text.count(kw) for kw in keywords])
                if matches > max_matches:
                    max_matches = matches
                    best_topic = topic_name
            
            topic_labels.append(f'{best_topic} ({len(cluster_papers)})')
        
        # create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=collab_matrix,
            x=topic_labels,
            y=topic_labels,
            colorscale='Purples',
            text=collab_matrix.astype(int),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='From: %{y}<br>To: %{x}<br>Overlapping Papers: %{z}<extra></extra>',
            colorbar=dict(title="Interdisciplinary<br>Papers")
        ))
        
        fig.update_layout(
            title='Cross-Topic Research Collaboration Intensity',
            xaxis_title='Research Topic',
            yaxis_title='Research Topic',
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='white'),
            height=600,
            xaxis={'tickangle': -45}
        )
        
        return fig
    
    def _create_keyword_cooccurrence_heatmap(self):
        print("computing keyword co-occurrence patterns...")
        
        # define topic keywords for co-occurrence analysis
        topics = {
            'Microgravity': ['microgravity', 'weightless', 'zero-g'],
            'Radiation': ['radiation', 'cosmic', 'ionizing'],
            'Bone Health': ['bone', 'skeletal', 'osteo'],
            'Plant Growth': ['plant', 'vegetation', 'botanical'],
            'Immunology': ['immune', 'immunity'],
            'Cardiovascular': ['cardiovascular', 'heart', 'vascular'],
            'Neuroscience': ['neuro', 'brain', 'cognitive'],
            'Cell Biology': ['cell', 'cellular', 'molecular']
        }
        
        topic_names = list(topics.keys())
        n_topics = len(topic_names)
        
        # initialize co-occurrence matrix
        cooccur_matrix = np.zeros((n_topics, n_topics))
        
        # for each paper, check which topics it mentions
        for idx, row in self.df.iterrows():
            # combine title and abstract for keyword matching
            text = f"{row['Title']} {row.get('abstract', '')}".lower()
            
            topic_present = []
            for i, (topic_name, keywords) in enumerate(topics.items()):
                if any(kw in text for kw in keywords):
                    topic_present.append(i)
            
            # update co-occurrence matrix
            for i in topic_present:
                for j in topic_present:
                    if i != j:  # don't count self-occurrence
                        cooccur_matrix[i, j] += 1
        
        # make symmetric
        cooccur_matrix = (cooccur_matrix + cooccur_matrix.T) / 2
        
        # create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cooccur_matrix,
            x=topic_names,
            y=topic_names,
            colorscale='Purples',
            text=cooccur_matrix.astype(int),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='Topic 1: %{y}<br>Topic 2: %{x}<br>Co-occurrences: %{z}<extra></extra>',
            colorbar=dict(title="Papers<br>Mentioning<br>Both Topics")
        ))
        
        fig.update_layout(
            title='Research Topic Co-occurrence in Publications',
            xaxis_title='Research Topic',
            yaxis_title='Research Topic',
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            font=dict(color='white'),
            height=600,
            xaxis={'tickangle': -45}
        )
        
        return fig
    
    def _create_placeholder_viz(self):
        # placeholder visualization when embeddings are not available
        fig = go.Figure()
        fig.add_annotation(
            text="Generate embeddings first to see 3D visualization",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color='white')
        )
        fig.update_layout(
            paper_bgcolor='#1e1e1e',
            plot_bgcolor='#1e1e1e',
            height=700
        )
        return fig