# utils.py
import os
import numpy as np
from extractor import is_valid_pe_with_code, extract_pe_header, construct_adjacency_matrix, power_iteration

MAP_LABELS = {
    "Benign": 0,
    "Backdoor": 1,
    "Exploit": 2,
    "Worm": 3,
    "Trojan": 4,
    "Ransomware": 5
}

def process_dataset_folder(folder_path):
    """Đường ống tự động biến đổi tệp thô thành các mảng vector đặc trưng trực tiếp trên RAM"""
    features_list = []
    
    if not os.path.exists(folder_path):
        print(f"[!] Thư mục không tồn tại: {folder_path}")
        return np.array([])

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            # Bộ lọc kép: Phải là PE hợp lệ VÀ phải chứa đoạn mã thực thi để chạy thuật toán mã hóa
            if not is_valid_pe_with_code(file_path):
                continue

            header_bytes = extract_pe_header(file_path)
            if header_bytes:
                adj_matrix = construct_adjacency_matrix(header_bytes)
                feature_vector = power_iteration(adj_matrix)
                features_list.append(feature_vector)
                
    return np.array(features_list)


def process_dataset_folder_advanced(folder_path, label):
    """
    Quét thư mục và trả về danh sách các đặc trưng nâng cao với nhãn số được chỉ định:
    - feature_vector (256-d) cho ML/MLP
    - raw_header (1024-d) cho CNN
    - edge_index, edge_weight cho GNN
    - label (int) nhãn lớp số của thư mục
    """
    data_list = []
    
    if not os.path.exists(folder_path):
        print(f"[!] Thư mục không tồn tại: {folder_path}")
        return data_list

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            if not is_valid_pe_with_code(file_path):
                continue
                
            header_bytes = extract_pe_header(file_path)
            if not header_bytes:
                continue
                
            # 1. Trích xuất đặc trưng cơ bản (vector 256-d)
            adj_matrix = construct_adjacency_matrix(header_bytes)
            feature_vector = power_iteration(adj_matrix)
            
            # 2. Định dạng cho CNN (chuỗi byte chuẩn hóa từ 0.0 - 1.0)
            raw_header = np.frombuffer(header_bytes, dtype=np.uint8).astype(np.float32) / 255.0
            
            # 3. Định dạng cho GNN (đồ thị)
            edge_indices = np.argwhere(adj_matrix > 0).T
            if edge_indices.size == 0:
                edge_index = np.empty((2, 0), dtype=np.int64)
                edge_weight = np.empty((0,), dtype=np.float32)
            else:
                edge_index = edge_indices.astype(np.int64)
                edge_weight = adj_matrix[edge_indices[0], edge_indices[1]].astype(np.float32)
            
            data_list.append({
                'feature_vector': feature_vector,
                'raw_header': raw_header,
                'edge_index': edge_index,
                'edge_weight': edge_weight,
                'label': label,
                'file_name': file_name
            })
            
    return data_list