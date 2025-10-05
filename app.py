import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.embeddings import EmbeddingEngine
from src.knowledge_graph import KnowledgeGraph
from src.search_engine import HybridSearchEngine
from src.claim_extractor import ClaimExtractor
from src.visualizer import Visualizer
from src.multi_agent import MultiAgentSystem

st.set_page_config(
    page_title="NASA Space Biology Research Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: rgba(102, 126, 234, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .stButton>button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# streamlit doesn't persist state between reruns, need manual session tracking
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'embeddings_ready' not in st.session_state:
    st.session_state.embeddings_ready = False
if 'kg_built' not in st.session_state:
    st.session_state.kg_built = False

with st.sidebar:
    st.markdown("### 🛠️ System Controls")
    
    st.markdown("#### 1️⃣ Data Pipeline")
    if st.button("📥 Load Publications", use_container_width=True):
        with st.spinner("fetching 608 nasa publications..."):
            loader = DataLoader()
            df = loader.load_publications()
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success(f"loaded {len(df)} papers!")
    
    if st.session_state.data_loaded:
        st.markdown("#### 2️⃣ Generate Embeddings")
        model_choice = st.selectbox(
            "pick your model",
            ["BioLinkBERT-base", "SciBERT", "all-MiniLM-L6-v2"]
        )
        
        if st.button("🧠 Create Embeddings", use_container_width=True):
            with st.spinner("generating embeddings with " + model_choice):
                engine = EmbeddingEngine(model_name=model_choice)
                st.session_state.embeddings = engine.generate_embeddings(st.session_state.df)
                st.session_state.embedding_engine = engine
                st.session_state.embeddings_ready = True
                
                # FIX: Clear cached search engine when embeddings change
                if 'search_engine' in st.session_state:
                    del st.session_state.search_engine
                
                st.success("embeddings ready!")
    
    if st.session_state.embeddings_ready:
        st.markdown("#### 3️⃣ Build Knowledge Graph")
        if st.button("🕸️ Create KG + GNN", use_container_width=True):
            with st.spinner("building knowledge graph with gnn..."):
                kg = KnowledgeGraph(st.session_state.df, st.session_state.embeddings)
                kg.build_graph()
                st.session_state.kg = kg
                st.session_state.kg_built = True
                st.success("knowledge graph built!")
    
    if st.session_state.kg_built:
        st.markdown("#### 4️⃣ Extract Claims")
        if st.button("🔬 Extract Scientific Claims", use_container_width=True):
            with st.spinner("extracting claims with sciBERT..."):
                claim_ext = ClaimExtractor()
                claims = claim_ext.extract_claims(st.session_state.df)
                st.session_state.claims = claims
                st.success(f"extracted {len(claims)} claims!")
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    search_k = st.slider("search results", 5, 50, 20)
    st.session_state.search_k = search_k

st.markdown('<h1 class="main-header">🚀 NASA Space Biology Research Explorer</h1>', unsafe_allow_html=True)

if not st.session_state.data_loaded:
    st.info("👈 start by loading publications from the sidebar")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Publications", "608")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Years Covered", "2010-2024")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Research Areas", "Multi-domain")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Hybrid Search", 
        "🌍 3D Research Landscape", 
        "📊 Knowledge Graph",
        "⚡ Claim Analysis",
        "🔥 Heatmaps & Insights"
    ])
    
    with tab1:
        st.markdown("### 🔍 Hybrid Search System")
        st.caption("combines dense retrieval (embeddings) + sparse retrieval (bm25) + re-ranking")
        
        if st.session_state.embeddings_ready:
            # search engine is expensive to recreate, cache it in session
            if 'search_engine' not in st.session_state:
                search_engine = HybridSearchEngine(
                    st.session_state.df,
                    st.session_state.embeddings,
                    st.session_state.embedding_engine
                )
                st.session_state.search_engine = search_engine
            
            query = st.text_input("enter your research question:", 
                                placeholder="e.g., effects of microgravity on bone density")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                use_query_expansion = st.checkbox("query expansion", value=True)
            with col2:
                search_button = st.button("🚀 Search", use_container_width=True)
            
            if search_button and query:
                with st.spinner("searching across 608 papers..."):
                    results = st.session_state.search_engine.hybrid_search(
                        query,
                        k=st.session_state.search_k,
                        query_expansion=use_query_expansion
                    )
                    
                    st.markdown(f"### Found {len(results)} relevant papers")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"#{i} - {result['title'][:100]}... (score: {result['score']:.3f})"):
                            st.markdown(f"**Title:** {result['title']}")
                            st.markdown(f"**Relevance Score:** {result['score']:.4f}")
                            st.markdown(f"**Abstract Preview:** {result['abstract'][:300]}...")
                            if result.get('link'):
                                st.markdown(f"[📄 Read Full Paper]({result['link']})")
        else:
            st.warning("generate embeddings first from the sidebar!")
    
    with tab2:
        st.markdown("### 🌍 Interactive 3D Research Landscape")
        st.caption("umap projection of all papers - cluster = similar topics")
        
        if st.session_state.embeddings_ready:
            viz = Visualizer(st.session_state.df, st.session_state.embeddings)
            
            col1, col2 = st.columns([3, 1])
            with col2:
                color_by = st.selectbox("color by:", ["cluster", "year", "density"])
                show_time_slider = st.checkbox("time evolution", value=False)
            
            fig = viz.create_3d_landscape(color_by=color_by)
            st.plotly_chart(fig, use_container_width=True)
            
            # research gaps analysis
            st.markdown("### 🏔️ Research Mountains vs Valleys")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🏔️ Well-Studied Areas (Mountains)**")
                mountains = viz.identify_mountains()
                for topic in mountains[:5]:
                    st.success(f"• {topic}")
            with col2:
                st.markdown("**🏜️ Research Gaps (Valleys)**")
                valleys = viz.identify_valleys()
                for gap in valleys[:5]:
                    st.error(f"• {gap}")
        else:
            st.warning("generate embeddings first!")
    
    with tab3:
        st.markdown("### 📊 Knowledge Graph Visualization")
        st.caption("entities connected by citations and semantic similarity")
        
        if st.session_state.kg_built:
            kg = st.session_state.kg
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nodes", kg.num_nodes())
            with col2:
                st.metric("Edges", kg.num_edges())
            with col3:
                st.metric("Clusters", kg.num_communities())
            
            st.markdown("#### Network Visualization")
            graph_html = kg.create_interactive_graph()
            st.components.v1.html(graph_html, height=600, scrolling=True)
            
            st.markdown("#### 🤖 GNN-Based Insights")
            node_importance = kg.compute_node_importance()
            
            st.markdown("**Most Influential Papers:**")
            for paper, score in node_importance[:10]:
                st.markdown(f"• {paper} (importance: {score:.3f})")
        else:
            st.warning("build knowledge graph first!")
    
    with tab4:
        st.markdown("### ⚡ Scientific Claim Analysis")
        st.caption("automated claim extraction + conflict detection")
        
        if 'claims' in st.session_state:
            claims = st.session_state.claims
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Claims", len(claims))
            with col2:
                conflicts = sum(1 for c in claims if c.get('has_conflict'))
                st.metric("Conflicting Claims", conflicts)
            with col3:
                st.metric("Consensus Claims", len(claims) - conflicts)
            
            st.markdown("#### Evidence Strength Distribution")
            strength_data = pd.DataFrame([
                {'Strength': c.get('evidence_strength', 'unknown'), 'Count': 1} 
                for c in claims
            ]).groupby('Strength').count().reset_index()
            
            # hardcoded colors match our design system
            fig = px.bar(strength_data, x='Strength', y='Count',
                        color='Strength',
                        color_discrete_map={
                            'strong': '#00cc00',
                            'moderate': '#ffaa00',
                            'weak': '#ff4444'
                        })
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 🔴 Detected Conflicts")
            conflicting = [c for c in claims if c.get('has_conflict')]
            
            # limit to 10 to avoid ui clutter
            for conflict in conflicting[:10]:
                with st.expander(f"Conflict: {conflict['claim'][:80]}..."):
                    st.markdown(f"**Claim:** {conflict['claim']}")
                    st.markdown(f"**Conflicting with:** {conflict.get('conflicts_with', 'N/A')}")
                    st.markdown(f"**Evidence Strength:** {conflict['evidence_strength']}")
        else:
            st.warning("extract claims first from sidebar!")
    
    with tab5:
        st.markdown("### 🔥 Heatmaps & Research Insights")
        
        if st.session_state.data_loaded:
            # embeddings are optional for some visualizations
            viz = Visualizer(st.session_state.df, 
                           st.session_state.embeddings if st.session_state.embeddings_ready else None)
            
            st.markdown("#### Topic Evolution Over Time")
            heatmap_fig = viz.create_topic_year_heatmap()
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
            st.markdown("#### Author Collaboration Network")
            collab_fig = viz.create_collaboration_heatmap()
            st.plotly_chart(collab_fig, use_container_width=True)
            
            if st.button("🤖 Generate Multi-Agent Analysis"):
                with st.spinner("running multi-agent system..."):
                    agent_system = MultiAgentSystem(st.session_state.df)
                    insights = agent_system.analyze()
                    
                    st.markdown("### 🎯 AI-Generated Insights")
                    for insight in insights:
                        st.info(insight)
        else:
            st.warning("load data first!")

st.markdown("---")
st.markdown("**Built for NASA Space Biology Challenge** | Powered by BioLinkBERT, PyTorch Geometric, UMAP")