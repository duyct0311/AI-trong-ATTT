# collect_benign.py
import os
import shutil
import pefile

def auto_collect_benign_balanced(exe_limit=1000, dll_limit=1000, mui_limit=1000):
    """
    Tự động thu thập tệp sạch theo tỉ lệ cấu trúc nghiêm ngặt:
    Quét đa thư mục để gom đủ 1000 EXE, 1000 DLL, và 1000 MUI.
    """
    # MỞ RỘNG: Thêm SysWOW64 để vét cạn các file EXE hệ thống còn thiếu
    SOURCE_DIRS = [r"C:\Windows\System32", r"C:\Windows\SysWOW64"]
    TARGET_DIR = os.path.join("data", "benign")
    
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        
    counts = {"exe": 0, "dll": 0, "mui": 0}
    print("[+] Bắt đầu quét tự động tìm file sạch...")

    for source_dir in SOURCE_DIRS:
        if not os.path.exists(source_dir):
            continue
            
        print(f"[+] Đang quét tại: {source_dir}")
        
        for root, dirs, files in os.walk(source_dir):
            # Kiểm tra nếu tất cả các nhóm đã đạt chỉ tiêu thì thoát sớm
            if counts["exe"] >= exe_limit and counts["dll"] >= dll_limit and counts["mui"] >= mui_limit:
                break
                
            for file in files:
                file_lower = file.lower()
                source_file_path = os.path.join(root, file)
                
                # Phân loại loại tệp
                file_type = None
                if file_lower.endswith('.mui'):
                    file_type = "mui"
                elif file_lower.endswith('.exe'):
                    file_type = "exe"
                elif file_lower.endswith('.dll'):
                    file_type = "dll"
                    
                # Nếu không khớp hoặc nhóm đó đã đủ số lượng thì bỏ qua
                if not file_type or counts[file_type] >= (exe_limit if file_type == "exe" else dll_limit if file_type == "dll" else mui_limit):
                    continue
                    
                # Tạo tên tệp độc nhất dựa trên thư mục cha để tránh ghi đè trùng tên
                unique_name = f"{os.path.basename(source_dir)}_{os.path.basename(root)}___{file}"
                target_file_path = os.path.join(TARGET_DIR, unique_name)
                
                if os.path.exists(target_file_path):
                    continue
                    
                try:
                    # Xác thực cấu trúc PE nhanh trước khi copy (trừ tệp .mui)
                    if file_type != "mui":
                        pe = pefile.PE(source_file_path, fast_load=True)
                        pe.close()
                    
                    # Sao chép tệp vào kho dữ liệu
                    shutil.copy2(source_file_path, target_file_path)
                    counts[file_type] += 1
                    
                    if sum(counts.values()) % 200 == 0:
                        print(f" -> Tiến độ: EXE ({counts['exe']}/{exe_limit}) | DLL ({counts['dll']}/{dll_limit}) | MUI ({counts['mui']}/{mui_limit})")
                except:
                    continue
                    
    print(f"\n[+] HOÀN THÀNH THU THẬP TẬP SẠCH:")
    print(f" - Tổng số tệp thực tế thu được: {sum(counts.values())}")
    print(f" - Chi tiết: EXE: {counts['exe']} | DLL: {counts['dll']} | MUI: {counts['mui']}")

if __name__ == "__main__":
    auto_collect_benign_balanced(exe_limit=1000, dll_limit=1000, mui_limit=1000)