import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import pandas as pd
import os

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GCN Model Definition (Must match the one in gnn_cora.py)
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

def train():
    print(f"Starting training on {device}...")
    
    # Load Dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    
    # Initialize Model
    model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    history = []
    epochs = 200
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
            acc = int(correct) / int(data.val_mask.sum())
            history.append({"Epoch": epoch, "Loss": float(loss), "Val Acc": acc})
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Loss: {loss:.4f}, Val Acc: {acc:.4f}")

    # Save Model and History
    torch.save(model.state_dict(), 'gnn_model.pt')
    pd.DataFrame(history).to_csv('training_history.csv', index=False)
    print("Training finished! Model saved to 'gnn_model.pt' and history to 'training_history.csv'.")

if __name__ == "__main__":
    train()
