import streamlit as st
import os

# Fix for OMP: Error #15 (Multiple libiomp5md.dll initializations)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from social_network_app import show_social_network_page
from gnn_cora import show_gnn_cora_page

st.set_page_config(page_title="GNN Social Network Analyzer", layout="wide")

st.sidebar.info("Experiment No. 9: Graph Representation & GNN")

page = st.sidebar.radio("Navigate Tasks", 
                         ["Part A: Social Network Matrices", 
                          "Part B: Cora GNN Embeddings"])

if page == "Part A: Social Network Matrices":
    show_social_network_page()
else:
    show_gnn_cora_page()

# Footer
st.sidebar.divider()
st.sidebar.markdown("""
**Experiment Objectives:**
1. Identify Nodes & Edges
2. Recognize Node/Edge features
3. Compute Matrices (A, D, M, L)
4. Node Embeddings via GNN
""")
