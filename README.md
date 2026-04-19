An interactive Graph Neural Network (GNN) workbench and Streamlit dashboard designed to illustrate graph representations, compute fundamental graph matrices (Adjacency, Degree, Incidence, Laplacian), and visualize node embeddings using the Cora citation dataset.

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
