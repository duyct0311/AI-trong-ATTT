# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron phân loại đặc trưng 256 chiều"""
    def __init__(self, in_dim=256, num_classes=6):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN1DClassifier(nn.Module):
    """1D CNN phân loại chuỗi 1024 byte của PE Header"""
    def __init__(self, in_len=1024, num_classes=6):
        super(CNN1DClassifier, self).__init__()
        # Input shape: (Batch, 1, 1024)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=8, stride=4) # Out: (Batch, 16, 255)
        self.pool1 = nn.MaxPool1d(2)                          # Out: (Batch, 16, 127)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=4, stride=2)# Out: (Batch, 32, 62)
        self.pool2 = nn.MaxPool1d(2)                          # Out: (Batch, 32, 31)
        
        self.fc1 = nn.Linear(32 * 31, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Đảm bảo x có shape (Batch, 1, 1024)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CustomGCNLayer(nn.Module):
    """GCN Layer tự định nghĩa bằng PyTorch thuần để tối đa hóa khả năng tương thích hệ điều hành"""
    def __init__(self, in_features, out_features):
        super(CustomGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, edge_index, edge_weight=None):
        # x: (N, in_features) với N=256
        # edge_index: numpy hoặc tensor shape (2, E)
        # edge_weight: numpy hoặc tensor shape (E,)
        N = x.size(0)
        device = x.device
        
        # 1. Tạo ma trận kề từ edge_index
        adj = torch.zeros(N, N, device=device)
        if edge_index is not None and len(edge_index) > 0 and edge_index.shape[1] > 0:
            # Chuyển đổi sang tensor nếu là numpy
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
            if edge_weight is not None and not isinstance(edge_weight, torch.Tensor):
                edge_weight = torch.tensor(edge_weight, dtype=torch.float, device=device)
                
            adj[edge_index[0], edge_index[1]] = edge_weight if edge_weight is not None else 1.0
                
        # 2. Thêm self-loops (A_tilde = A + I)
        adj = adj + torch.eye(N, device=device)
        
        # 3. Tính toán ma trận chuẩn hóa đối xứng (D^-1/2 * A_tilde * D^-1/2)
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        norm_adj = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
        
        # 4. Tích chập đồ thị
        out = torch.mm(norm_adj, x)
        out = self.linear(out) + self.bias
        return out


class GraphAutoencoder(nn.Module):
    """Mô hình GAE phục vụ phát hiện bất thường"""
    def __init__(self, node_dim=256, hidden_dim=64, latent_dim=16):
        super(GraphAutoencoder, self).__init__()
        # Encoder
        self.gcn1 = CustomGCNLayer(node_dim, hidden_dim)
        self.gcn2 = CustomGCNLayer(hidden_dim, latent_dim)
        
    def encode(self, x, edge_index, edge_weight=None):
        h = F.relu(self.gcn1(x, edge_index, edge_weight))
        z = self.gcn2(h, edge_index, edge_weight)
        return z
        
    def decode(self, z):
        # Tái tạo ma trận kề bằng tích vô hướng
        adj_reconstructed = torch.sigmoid(torch.mm(z, z.t()))
        return adj_reconstructed
        
    def forward(self, x, edge_index, edge_weight=None):
        z = self.encode(x, edge_index, edge_weight)
        adj_rec = self.decode(z)
        return adj_rec, z


class GNNClassifier(nn.Module):
    """Mô hình phân loại đồ thị GNN đa lớp"""
    def __init__(self, node_dim=256, hidden_dim=64, num_classes=6):
        super(GNNClassifier, self).__init__()
        self.gcn1 = CustomGCNLayer(node_dim, hidden_dim)
        self.gcn2 = CustomGCNLayer(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x, edge_index, edge_weight=None):
        h = F.relu(self.gcn1(x, edge_index, edge_weight))
        h = F.relu(self.gcn2(h, edge_index, edge_weight))
        
        # Global Mean Pooling: Lấy trung bình biểu diễn của 256 nút
        hg = torch.mean(h, dim=0, keepdim=True) # Out: (1, hidden_dim)
        
        x_out = F.relu(self.fc1(hg))
        x_out = self.fc2(x_out)
        return x_out
