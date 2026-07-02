# main.py
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import joblib
import pefile
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from extractor import extract_pe_header, construct_adjacency_matrix, power_iteration
from models import MLPClassifier, CNN1DClassifier, GraphAutoencoder, GNNClassifier

MAP_LABELS_REV = {
    0: "Benign (An toàn/Sạch)",
    1: "Backdoor (Mã độc cửa sau)",
    2: "Exploit (Mã khai thác lỗ hổng)",
    3: "Worm (Sâu máy tính)",
    4: "Trojan (Mã độc Trojan)",
    5: "Ransomware (Mã độc tống tiền)"
}

# Thiết lập thiết bị tính toán (CUDA nếu có)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models():
    """Tải tất cả các mô hình có sẵn vào bộ nhớ để quét đối chứng"""
    models_dict = {}
    MODEL_DIR = "model"
    
    # 1. Random Forest
    rf_path = os.path.join(MODEL_DIR, "ransomware_rf_model.pkl")
    if os.path.exists(rf_path):
        models_dict["rf"] = joblib.load(rf_path)
        
    # 2. Decision Tree
    dt_path = os.path.join(MODEL_DIR, "decision_tree_model.pkl")
    if os.path.exists(dt_path):
        models_dict["dt"] = joblib.load(dt_path)
        
    # 3. XGBoost
    xgb_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    if os.path.exists(xgb_path):
        models_dict["xgb"] = joblib.load(xgb_path)
        
    # 3.5. SVM
    svm_path = os.path.join(MODEL_DIR, "svm_model.pkl")
    if os.path.exists(svm_path):
        models_dict["svm"] = joblib.load(svm_path)
        
    # 4. MLP
    mlp_path = os.path.join(MODEL_DIR, "mlp_model.pt")
    if os.path.exists(mlp_path):
        mlp = MLPClassifier(num_classes=6)
        try:
            mlp.load_state_dict(torch.load(mlp_path, map_location=DEVICE))
            mlp.to(DEVICE)
            mlp.eval()
            models_dict["mlp"] = mlp
        except Exception as e:
            print(f"[-] Lỗi nạp mô hình MLP: {e}")
            
    # 5. 1D-CNN
    cnn_path = os.path.join(MODEL_DIR, "cnn_model.pt")
    if os.path.exists(cnn_path):
        cnn = CNN1DClassifier(num_classes=6)
        try:
            cnn.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
            cnn.to(DEVICE)
            cnn.eval()
            models_dict["cnn"] = cnn
        except Exception as e:
            print(f"[-] Lỗi nạp mô hình 1D-CNN: {e}")
            
    # 6. GNN Classifier
    gnn_path = os.path.join(MODEL_DIR, "gnn_classifier_model.pt")
    if os.path.exists(gnn_path):
        gnn = GNNClassifier(num_classes=6)
        try:
            gnn.load_state_dict(torch.load(gnn_path, map_location=DEVICE))
            gnn.to(DEVICE)
            gnn.eval()
            models_dict["gnn"] = gnn
        except Exception as e:
            print(f"[-] Lỗi nạp mô hình GNN Classifier: {e}")
            
    # 7. Graph Autoencoder (GAE)
    gae_path = os.path.join(MODEL_DIR, "gae_anomaly_model.pt")
    gae_meta_path = os.path.join(MODEL_DIR, "gae_metadata.pkl")
    if os.path.exists(gae_path) and os.path.exists(gae_meta_path):
        gae = GraphAutoencoder()
        try:
            gae.load_state_dict(torch.load(gae_path, map_location=DEVICE))
            gae.to(DEVICE)
            gae.eval()
            models_dict["gae"] = gae
            models_dict["gae_threshold"] = joblib.load(gae_meta_path)["threshold"]
        except Exception as e:
            print(f"[-] Lỗi nạp mô hình GAE: {e}")
            
    return models_dict

def scan_file(file_path, models_dict):
    """Phân tích tĩnh một tệp thực thi chỉ định trên nhiều mô hình để đối chứng"""
    print(f"\n[+] Đang tiến hành quét phân tích tĩnh: {file_path}")
    
    # 1. Kiểm tra sự tồn tại của tệp
    if not os.path.exists(file_path):
        print("[-] Lỗi: Đường dẫn tệp tin không tồn tại.")
        return
        
    # 2. Kiểm tra định dạng Windows PE hợp lệ
    try:
        pe = pefile.PE(file_path, fast_load=True)
        pe.close()
    except pefile.PEFormatError:
        print("[!] Kết quả: Bỏ qua phân tích. Tệp tin không thuộc định dạng cấu trúc Windows PE.")
        return

    # 3. Trích xuất đặc trưng
    header_bytes = extract_pe_header(file_path)
    if not header_bytes:
        print("[-] Lỗi: Không thể trích xuất cấu trúc byte từ Header của tệp.")
        return
        
    adj_matrix = construct_adjacency_matrix(header_bytes)
    feature_vector = power_iteration(adj_matrix)
    
    # CNN format
    raw_header = np.frombuffer(header_bytes, dtype=np.uint8).astype(np.float32) / 255.0
    
    # GNN format
    edge_indices = np.argwhere(adj_matrix > 0).T
    if edge_indices.size == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_weight = np.empty((0,), dtype=np.float32)
    else:
        edge_index = edge_indices.astype(np.int64)
        edge_weight = adj_matrix[edge_indices[0], edge_indices[1]].astype(np.float32)
        
    # 4. Dự đoán
    print("================ KẾT QUẢ PHÂN TÍCH TĨNH ================")
    
    # --- Random Forest ---
    if "rf" in models_dict:
        pred = models_dict["rf"].predict(feature_vector.reshape(1, -1))[0]
        # Tương thích với cả mô hình cũ (2 lớp) và mô hình mới (6 lớp)
        if hasattr(models_dict["rf"], "classes_") and len(models_dict["rf"].classes_) == 6:
            label_str = MAP_LABELS_REV.get(pred, "Không xác định")
        else:
            label_str = "CẢNH BÁO: RANSOMWARE" if pred == 1 else "Benign (An toàn)"
        print(f"[*] [Random Forest]: {label_str}")
        
    # --- XGBoost ---
    if "xgb" in models_dict:
        pred = models_dict["xgb"].predict(feature_vector.reshape(1, -1))[0]
        print(f"[*] [XGBoost]: {MAP_LABELS_REV.get(pred, 'Không xác định')}")
        
    # --- Decision Tree ---
    if "dt" in models_dict:
        pred = models_dict["dt"].predict(feature_vector.reshape(1, -1))[0]
        print(f"[*] [Decision Tree]: {MAP_LABELS_REV.get(pred, 'Không xác định')}")
        
    # --- SVM ---
    if "svm" in models_dict:
        pred = models_dict["svm"].predict(feature_vector.reshape(1, -1))[0]
        print(f"[*] [SVM]: {MAP_LABELS_REV.get(pred, 'Không xác định')}")
        
    # --- MLP ---
    if "mlp" in models_dict:
        model = models_dict["mlp"]
        with torch.no_grad():
            feat_t = torch.tensor(feature_vector.reshape(1, -1), dtype=torch.float, device=DEVICE)
            out = model(feat_t)
            pred = torch.argmax(out, dim=1).item()
        print(f"[*] [MLP - PyTorch]: {MAP_LABELS_REV.get(pred, 'Không xác định')}")
        
    # --- 1D-CNN ---
    if "cnn" in models_dict:
        model = models_dict["cnn"]
        with torch.no_grad():
            feat_t = torch.tensor(raw_header.reshape(1, 1, -1), dtype=torch.float, device=DEVICE)
            out = model(feat_t)
            pred = torch.argmax(out, dim=1).item()
        print(f"[*] [1D-CNN - PyTorch]: {MAP_LABELS_REV.get(pred, 'Không xác định')}")
        
    # --- GNN Classifier ---
    if "gnn" in models_dict:
        model = models_dict["gnn"]
        x = torch.eye(256, dtype=torch.float, device=DEVICE)
        edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=DEVICE)
        edge_weight_t = torch.tensor(edge_weight, dtype=torch.float, device=DEVICE)
        with torch.no_grad():
            out = model(x, edge_index_t, edge_weight_t)
            pred = torch.argmax(out, dim=1).item()
        print(f"[*] [GNN Classifier]: {MAP_LABELS_REV.get(pred, 'Không xác định')}")
        
    # --- GAE Anomaly Detector ---
    if "gae" in models_dict:
        model = models_dict["gae"]
        threshold = models_dict["gae_threshold"]
        x = torch.eye(256, dtype=torch.float, device=DEVICE)
        edge_index_t = torch.tensor(edge_index, dtype=torch.long, device=DEVICE)
        edge_weight_t = torch.tensor(edge_weight, dtype=torch.float, device=DEVICE)
        with torch.no_grad():
            adj_rec, _ = model(x, edge_index_t, edge_weight_t)
            adj_gt = torch.zeros(256, 256, device=DEVICE)
            if edge_index.shape[1] > 0:
                adj_gt[edge_index[0], edge_index[1]] = edge_weight_t
            adj_gt = adj_gt + torch.eye(256, device=DEVICE)
            score = F.mse_loss(adj_rec, adj_gt).item()
            
        is_anomaly = score > threshold
        anomaly_str = "🔴 CẢNH BÁO BẤT THƯỜNG" if is_anomaly else "🟢 BÌNH THƯỜNG (An toàn)"
        print(f"[*] [GAE Anomaly] Score: {score:.6f} (Ngưỡng: {threshold:.6f}) -> {anomaly_str}")
        
    print("========================================================")

def main():
    print("[+] Đang nạp tất cả mô hình phát hiện mã độc...")
    models_dict = load_models()
    
    if not models_dict:
        print("[-] Cảnh báo: Chưa có mô hình nào được huấn luyện.")
        print("[!] Hãy đảm bảo bạn đã chạy 'python train_compare.py' trước để sinh file mô hình.")
        sys.exit(1)
        
    print(f"[+] Đã nạp thành công {len(models_dict)} mô hình quét. Engine Scanner sẵn sàng.")
    print("Gõ 'exit' để thoát chương trình.")
    
    while True:
        target_file = input("\nNhập đường dẫn tệp cần quét (hoặc kéo thả tệp vào đây): ").strip()
        target_file = target_file.strip('"').strip("'")
        
        if target_file.lower() == 'exit':
            print("[+] Đóng hệ thống phân tích tĩnh.")
            break
            
        if not target_file:
            continue
            
        scan_file(target_file, models_dict)

if __name__ == "__main__":
    main()