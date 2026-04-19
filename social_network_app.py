import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
from utils import calculate_matrices, plot_graph_matplotlib

def show_social_network_page():
    st.header("Graph Representation for Social Network")
    st.write("### Experiment Part A: Illustrate and Compute Graph Matrices")
    
    # Selection: Real or Synthetic
    data_source = st.selectbox("Select Social Network Source", 
                               ["Synthetic Small Network (5 Nodes)", 
                                "Karaté Club Dataset (Famous Social Graph)", 
                                "Simulated Facebook Ego-Network"])
    
    if data_source == "Synthetic Small Network (5 Nodes)":
        G = nx.Graph()
        G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)])
        node_features = {
            1: "Admin", 2: "User", 3: "Moderator", 4: "User", 5: "VIP"
        }
        nx.set_node_attributes(G, node_features, "Role")
    elif data_source == "Karaté Club Dataset (Famous Social Graph)":
        G = nx.karate_club_graph()
    else:
        # Simulated Ego-Facebook
        G = nx.powerlaw_cluster_graph(50, 2, 0.1, seed=42)

    st.write(f"**Network Stats:** {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Graph Visualization")
        fig = plot_graph_matplotlib(G, title=f"Social Network View ({data_source})")
        st.pyplot(fig)
        
    with col2:
        st.subheader("Nodes & Features")
        node_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
        st.dataframe(node_df if not node_df.empty else pd.DataFrame({"Node ID": list(G.nodes())}))

    st.divider()
    st.subheader("Compute Graph Matrices")
    
    # Calculate Matrices
    A, D, L, M, nodes = calculate_matrices(G)
    
    # Display in Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Adjacency Matrix (A)", 
        "Degree Matrix (D)", 
        "Laplacian Matrix (L)", 
        "Incidence Matrix (M)"
    ])
    
    with tab1:
        st.write("Describes connection between nodes. $A_{ij}=1$ if there is an edge.")
        st.dataframe(pd.DataFrame(A, index=nodes, columns=nodes))
        
    with tab2:
        st.write("Diagonal matrix containing the degree of each node.")
        st.dataframe(pd.DataFrame(D, index=nodes, columns=nodes))
        
    with tab3:
        st.write("Measure of network smoothness: $L = D - A$. Row sums are always 0.")
        st.dataframe(pd.DataFrame(L, index=nodes, columns=nodes))
        
    with tab4:
        st.write("Relationship between nodes (rows) and edges (columns).")
        st.dataframe(pd.DataFrame(M, index=nodes))

if __name__ == "__main__":
    show_social_network_page()
