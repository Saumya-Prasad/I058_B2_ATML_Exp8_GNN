import streamlit as st
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from utils import plot_graph_matplotlib, get_comp_graph_data
import numpy as np
import pandas as pd

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simple GCN Model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

@st.cache_resource
def load_cora():
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
    return dataset

def show_gnn_cora_page():
    st.header("GNN Analysis Dashboard (Pre-trained)")
    st.write(f"**Current Device:** {device}")
    st.write("### Experiment Part B: GCN with Cora Dataset")
    
    with st.spinner("Loading Cora Dataset..."):
        dataset = load_cora()
        data = dataset[0].to(device)
        
    st.write(f"**Cora Stats:** {data.num_nodes} Nodes, {data.num_edges} Edges, {dataset.num_features} Features")
    
    # Initialize Model AND Load Weights
    model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
    
    model_path = 'gnn_model.pt'
    history_path = 'training_history.csv'
    
    if os.path.exists(model_path) and os.path.exists(history_path):
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Load History
        history_df = pd.read_csv(history_path)
        
        st.divider()
        st.subheader("Training Performance (Background Run)")
        st.write("The model was trained offline. Below are the metrics saved during the run.")
        st.line_chart(history_df.set_index("Epoch")[["Loss", "Val Acc"]])
        st.success(f"Model successfully loaded from `{model_path}`")
    else:
        st.error("Pre-trained model files not found. Please run `python train_gnn.py` first.")
        return

    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Select Node for Analysis")
        node_id = st.number_input("Enter Node ID (0-2707)", min_value=0, max_value=int(data.num_nodes)-1, value=0)
        num_hops = st.slider("Computational Graph Hops", 1, 2, 1)
        
        # Computational Graph Visualization
        sub_g, subset = get_comp_graph_data(data, node_id, num_hops)
        st.write(f"Showing {num_hops}-hop neighborhood (Computational Graph for Node {node_id})")
        fig = plot_graph_matplotlib(sub_g, title=f"Computational Graph (Node {node_id})")
        st.pyplot(fig)

    with col2:
        st.subheader("Node Embedding (Hidden Representation)")
        
        # Get Embedding
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            node_embedding = out[node_id].cpu().numpy()
            
        st.write("This is the output vector of the GNN for the selected node.")
        st.code(f"Embedding Shape: {node_embedding.shape}")
        
        # Plot as a heatmap or bar chart
        st.bar_chart(node_embedding)
        
        with st.expander("View Raw Embedding Vector"):
            st.write(node_embedding)

    st.subheader("Node Features (BoW representation)")
    features = data.x[node_id].cpu().numpy()
    non_zero_feats = np.where(features > 0)[0]
    st.write(f"Node {node_id} has {len(non_zero_feats)} active features (word indices).")
    st.json(non_zero_feats.tolist())

if __name__ == "__main__":
    show_gnn_cora_page()
