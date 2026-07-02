# extractor.py
import os
import numpy as np
import pefile

# Đoạn hàm cập nhật lại trong extractor.py cho mục tiêu học dữ liệu hỗn hợp:
def is_valid_pe_with_code(file_path):
    if ":" in os.path.basename(file_path):
        return False
    try:
        # Chỉ cần mở được cấu trúc PE chuẩn của Windows là cho đi qua để nạp vào RAM huấn luyện
        pe = pefile.PE(file_path, fast_load=True)
        pe.close()
        return True
    except:
        return False

def extract_pe_header(file_path):
    """Trích xuất đúng 1024 bytes đầu tiên của tệp thực thi (PE Header)"""
    try:
        with open(file_path, 'rb') as f:
            header_bytes = f.read(1024)
        if len(header_bytes) < 1024:
            header_bytes += b'\x00' * (1024 - len(header_bytes))
        return header_bytes
    except Exception as e:
        print(f"[-] Lỗi đọc cấu trúc tệp {file_path}: {e}")
        return None

def construct_adjacency_matrix(header_bytes):
    """Xây dựng ma trận kề đồ thị hướng 256x256 dựa trên Byte 2-gram (chưa chuẩn hóa)"""
    matrix = np.zeros((256, 256), dtype=np.float64)
    
    for i in range(len(header_bytes) - 1):
        byte_current = header_bytes[i]
        byte_next = header_bytes[i+1]
        matrix[byte_current, byte_next] += 1.0
        
    return matrix

def power_iteration(adj_matrix, max_iterations=100, tol=1e-6):
    """Thuật toán Power Iteration tìm vectơ riêng trội để nhúng đồ thị thành Vector 256 chiều
    Sử dụng công thức (5) của bài báo để khởi tạo v0 và tiến hành chuẩn hóa ma trận kề.
    """
    # 1. Tính toán vector khởi tạo v0 theo Công thức (5) của bài báo:
    # v0_i = (Tổng hàng i) / (Tổng cột i) của ma trận kề chưa chuẩn hóa
    row_sums = adj_matrix.sum(axis=1)
    column_sums = adj_matrix.sum(axis=0)
    
    # Tránh chia cho 0 khi tính toán v0
    column_sums_safe = np.where(column_sums == 0, 1.0, column_sums)
    v = row_sums / column_sums_safe
    
    # Chuẩn hóa v về vector đơn vị
    v_norm = np.linalg.norm(v)
    if v_norm > 0:
        v = v / v_norm
    else:
        # Nếu ma trận kề toàn số 0 (không có 2-gram nào), khởi tạo ngẫu nhiên
        v = np.random.rand(256)
        v = v / np.linalg.norm(v)
        
    # 2. Chuẩn hóa ma trận kề theo cột để thu được ma trận W
    column_sums[column_sums == 0] = 1.0  # Tránh lỗi chia cho 0
    W = adj_matrix / column_sums
    
    # 3. Lặp tìm vector riêng trội của ma trận W
    for _ in range(max_iterations):
        v_next = np.dot(W, v)
        v_next_norm = np.linalg.norm(v_next)
        if v_next_norm == 0:
            break
        v_next = v_next / v_next_norm
        
        if np.allclose(v, v_next, atol=tol):
            v = v_next
            break
        v = v_next
        
    return v