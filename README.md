<br>

## 📌 Streamlit Frontend Screenshots - Social Network

<img width="1892" height="816" alt="image" src="Streamlit Screenshot Frontend/image_main.png" />

**Dynamic Network Illustration:** Choose between synthetic datasets, the famous Karate Club network, or simulated Facebook Ego-networks to observe different graph topologies.

**Real-time Feature Mapping:** The dashboard automatically extracts and displays node-level features (e.g., Roles like Admin, Moderator, VIP) alongside the visual graph layout.

**Graph Inference:** 
- In the example synthetic network, **Nodes 2 and 3** act as central hubs with the highest degree of connectivity.
- The representation successfully maps **heterogeneous node features** (Roles) to the structural topology, a prerequisite for GNN message passing.
- The visualization confirms the existence of a high-connectivity **clique** between the primary admin and user nodes.

<br>

<img width="1892" height="816" alt="image" src="Streamlit Screenshot Frontend/image_facebook.png" />

**Large-Scale Social Simulation:** Explore a **Facebook Ego-Network** simulation (50 nodes, 96 edges), demonstrating the complexity of real-world social clusters and inter-group connectivity.

**Social Network Inference:** 
- The ego-network exhibits high **Local Clustering**, where tightly-knit groups of friends form discrete triangles, reflecting the "friends of friends" property of human interaction.
- Nodes positioned between major clusters act as **Social Bridges**, controlling the flow of information across otherwise isolated sub-communities.
- At an average degree of ~3.8, the network demonstrates a robust connectivity that allows for effective GNN feature propagation.

<br>

<img width="1892" height="816" alt="image" src="Streamlit Screenshot Frontend/image_copy_1.png" />

**Automated Matrix Engine:** The workbench instantly computes the Adjacency, Degree, Laplacian, and Incidence matrices, providing a mathematical "fingerprint" of the network.

**Mathematical Inference:** 
- The **Symmetry** of the Adjacency Matrix confirms the network is undirected ($A_{ij} = A_{ji}$).
- The **Zero-Diagonal** property indicates that the network contains no self-loops, maintaining a clean relational structure.
- The **Sparsity** of the entries reflects the "Small World" phenomenon, where nodes are interconnected through a limited number of high-influence hubs rather than dense, all-to-all links.

<br>

<img width="1892" height="816" alt="image" src="Streamlit Screenshot Frontend/image_copy_2.png" />

**Degree Distribution Mapping:** Displays the Degree Matrix (D), which summarizes the connectivity profile of every node in the graph.

**Mathematical Inference:** 
- The **Diagonal Entries** $D_{ii}$ coincide with the physical number of edges connected to each node, identifying **Nodes 2 and 3** as the primary hubs.
- In GNN theory, this matrix is critical for **Normalization**; it prevents high-degree nodes from overwhelming the feature updates during the neighborhood aggregation phase.

<br>

<img width="1892" height="816" alt="image" src="Streamlit Screenshot Frontend/image_laplacian.png" />

**Laplacian Smoothness Metric:** Shows the Laplacian Matrix (L), computed as $L = D - A$, representing the discrete "slope" or curvature of signals across the graph.

**Mathematical Inference:** 
- The **Row-Sum Zero Property** ($L\mathbf{1} = \mathbf{0}$) is mathematically verified, signifying that the matrix is positive semi-definite and capturing the diffusion properties of the network.
- The magnitude of the off-diagonal entries highlights the **strength of social influences** between connected pairs.
- In GNNs, the Laplacian is the foundation for **Spectral Convolutions**, allowing the model to perform signal processing (filtering) directly on the graph structure.

<br>

<img width="1892" height="816" alt="image" src="Streamlit Screenshot Frontend/image_incidence.png" />

**Node-Edge Mapping:** Displays the Incidence Matrix (M), illustrating which edges (columns) connect to which nodes (rows).

**Mathematical Inference:** 
- Every column in the Incidence matrix contains exactly **two non-zero entries**, perfectly identifying the start and end points of every relationship.
- This matrix serves as a fundamental building block for the Laplacian ($L = MM^T$), acting as the first-order derivative of the graph's topology.
- It is the primary tool for analyzing **flow and circulation** in networks, making it essential for complex traffic and communication simulations.

<br>

<img width="1892" height="816" alt="image" src="Streamlit Screenshot Frontend/image_copy_3.png" />

**Classic Network Benchmark:** Explore the **Zachary's Karate Club** dataset (34 nodes, 78 edges), a gold standard for testing community detection and graph partitioning algorithms.

**Social Network Inference:** 
- The dense web of edges reveals a clear **Bipolar Community Structure**, where nodes naturally cluster around two leaders (Node 0 and Node 33).
- **Factional Split**: The "Mr. Hi" vs. "Officer" labels illustrate how social ties dictate factional loyalty during a real-world organizational conflict.
- **Node Hubs**: High-degree nodes in this network act as information brokers, bridging the gap between smaller groups of students and the central leadership.

<br>

## 📌 Streamlit Frontend Screenshots - Cora GNN

<img width="1892" height="816" alt="image" src="Streamlit Screenshot Frontend/image_gnn_training.png" />

**GNN Training Benchmarks:** Real-time tracking of the Graph Convolutional Network's learning curve over 200 epochs on the Cora dataset.

**Model Inference:** 
- **Convergence characteristics**: The smooth decline in loss confirms that the **Adam optimizer** effectively traversed the high-dimensional error surface of the citation network.
- **Accuracy Plateau**: Reaching a peak validation accuracy of **~74%** demonstrates that the 2-layer GCN successfully integrated both the Bag-of-Words features and the citation topology.

<br>

<img width="1892" height="816" alt="image" src="Streamlit Screenshot Frontend/image_gnn_node_analysis.png" />

**Localized Message Passing (1-Hop):** Visualization of the immediate neighborhood used to compute Node 0's intermediate features.

**Inference & Logic:** 
- **Neighborhood Aggregation**: The computational graph shows the exact receptive field used to calculate the node's feature update, illustrating the core logic of **Graph convolutions**.
- **Embedding Space**: The 7-dimensional bar chart represents the paper's latent signature, mapping high-dimensional text data and structure into a dense vector for downstream classification.

<br>

<img width="1892" height="816" alt="image" src="Streamlit Screenshot Frontend/image_gnn_node_hops.png" />

**Multi-Hop Receptive Field (2-Hop):** Shows how the GNN expands its context to include "neighbors of neighbors," capturing complex structural patterns.

**Inference & Logic:** 
- **Exponential Expansion**: Notice how the computational graph grows. This demonstrates the **Depth in GNNs**, where each layer aggregates one additional hop of structural information.
- **Structural Context**: The 2-hop view captures subtle group dynamics that are invisible at the 1-hop level.

<br>

<img width="1892" height="816" alt="image" src="Streamlit Screenshot Frontend/image_gnn_node_features.png" />

**Input Feature Sparsity (BoW):** Detailed view of the Bag-of-Words word indices associated with paper Node 0.

**Inference & Logic:** 
- **The Sparse-to-Dense Journey**: Node 0 possesses 9 active features. The GNN transforms this extreme sparsity into a robust, dense semantic vector.
- **Feature Extraction**: These indices represent key scholarly terms within the paper's abstract.

<br>

<img width="1892" height="816" alt="image" src="Streamlit Screenshot Frontend/image_gnn_node_1215_analysis.png" />

**High-Degree Periphery Analysis (Node 1215):** Testing the model on a complex, high-connectivity node from a different scholarly topic.

**Inference & Logic:** 
- **Scalability**: For peripheral but highly connected nodes like 1215, the 2-hop computation involves a much larger subgraph, confirming the efficiency of the GNN's local sampling logic.
- **Topological Divergence**: The distinct embedding profile compared to Node 0 confirms that the GNN is highly sensitive to the unique structural position of each paper.

<br>

<img width="1892" height="816" alt="image" src="Streamlit Screenshot Frontend/image_gnn_node_1215_features.png" />

**Feature Robustness Check:** Node 1215 has only 5 active features, yet the model still generates a distinct, high-quality embedding signature.

**Inference & Logic:** 
- **Reliable Learning**: This demonstrates the GNN's **robustness to feature drop**; even with fewer input words, the model uses structural neighbors to compensate for "missing" information.

<br>

## 📌 Project Overview
Graphs are essential for modeling complex relationships in social networks, biology, and citation data. This project implements a full pipeline for graph analysis and GNN modeling to demonstrate the transition from static graph matrices to dynamic node embeddings.

**Focus Areas:**

🔹 **Social Network Analysis:** Illustrating connectivity and computing structural matrices.

🔹 **Node Embeddings:** Using Graph Convolutional Networks (GCN) to map high-dimensional citation data into a latent feature space.

<br>

## 📌 Graph Representation & Matrices
A graph is defined by its nodes (entities) and edges (relationships). In Part A of this experiment, we compute four critical matrices:

🔸 **Adjacency Matrix (A):** Captures the direct connectivity between nodes.

🔸 **Degree Matrix (D):** A diagonal matrix representing the number of connections for each node.

🔸 **Incidence Matrix (M):** Maps the relationship between nodes and edges.

🔸 **Laplacian Matrix (L):** Computed as $L = D - A$, this matrix is a measure of network smoothness and is crucial for graph spectral theory.

<br>

## 📌 Graph Neural Network (GNN) Implementation
Moving beyond static matrices, we implement a **Graph Convolutional Network (GCN)** to learn from the Cora citation network.

🔸 **Message Passing:** Nodes aggregate information from their local neighborhood to update their internal feature representation.

🔸 **Computational Graph:** Visualizing the 1-hop and 2-hop neighborhoods used by the GNN to calculate a specific node's embedding.

🔸 **Node Embedding:** The GNN transforms the Bag-of-Words (BoW) paper features into a dense vector (embedding) that preserves the graph's structural topology.

<br>

## 📌 Implementation Parameters
Based on standard GNN benchmarks, the model is trained offline to ensure stability and performance:

🔹 **Architecture:** 2-layer GCN (GraphConv) with ReLU activation and Dropout.

🔹 **Dataset:** Cora Citation Network (2,708 nodes, 5,429 edges).

🔹 **Optimizer:** Adam with $0.01$ learning rate and weight decay.

🔹 **Epochs:** **200**, reaching a peak validation accuracy of ~74.6%.

🔹 **Device:** Multi-device support (CUDA/CPU) with optimized OMP runtime handling.

<br>

## 📌 The Streamlit Dashboard
The results are presented in a unified Streamlit interface, allowing for interactive exploration:

🔸 **Matrix Explorer:** View and download structural matrices for synthetic and classic (Karate Club) social graphs.

🔸 **GNN Lab:** Select any Node ID to view its localized computational graph and inspect its resulting high-dimensional embedding vector.

🔸 **Training Progress:** Real-time visualization of the Loss and Accuracy curves from the 200-epoch training session.

<br>

## 📌 Tech Stack
🔹 **Deep Learning:** PyTorch, PyTorch Geometric (PyG)

🔹 **Frontend:** Streamlit

🔹 **Graph Analysis:** NetworkX

🔹 **Data Visualization:** Matplotlib, Pandas, NumPy

🔹 **Language:** Python

<br>

## 📥 How to Run Locally
🔸 Clone the repository to your local machine.

🔸 Ensure you have Python installed and create a virtual environment.

🔸 Install the requirements: `pip install -r requirements.txt`

🔸 Run the standalone training script: `python train_gnn.py`

🔸 Launch the Streamlit dashboard: `streamlit run app.py`
