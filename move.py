# move.py
import os
import shutil
from extractor import is_valid_pe_with_code

# Cấu hình nhãn tương ứng với thư mục lưu trữ
MALWARE_CLASSES = {
    "backdoor": "backdoor",
    "exploit": "exploit",
    "worm": "worm",
    "trojan": "trojan",
    "ransomware": "ransomware"
}

def get_malware_class(file_name):
    """Phân loại họ mã độc dựa trên tên tệp tin"""
    name_lower = file_name.lower()
    if "backdoor" in name_lower or "njrat" in name_lower or "bifrost" in name_lower or "asyncrat" in name_lower:
        return "backdoor"
    elif "exploit" in name_lower or "cve-" in name_lower or "cobaltstrike" in name_lower:
        return "exploit"
    elif "worm" in name_lower or "conficker" in name_lower or "mydoom" in name_lower or "jenaxa" in name_lower:
        return "worm"
    elif "trojan" in name_lower or "emotet" in name_lower or "agenttesla" in name_lower or "formbook" in name_lower:
        return "trojan"
    elif "ransomware" in name_lower or "wannacry" in name_lower or "locky" in name_lower or "cerber" in name_lower or "stop" in name_lower:
        return "ransomware"
    else:
        return None

def process_and_classify_dataset(source_dir, target_base_dir="data", limit_per_class=4000):
    """
    Quét thư mục chứa mẫu mã độc thô, lọc tệp PE hợp lệ,
    phân loại vào các thư mục riêng biệt tương ứng với từng họ mã độc
    và giới hạn số lượng tối đa 4000 tệp cho mỗi họ. Các tệp thừa hoặc không hợp lệ sẽ bị xóa.
    """
    print(f"[+] Bắt đầu quy trình quét và phân loại tệp từ: {source_dir}")
    print(f"[+] Ngưỡng thu thập tối đa: {limit_per_class} file/loại mã độc")
    
    if not os.path.exists(source_dir):
        print(f"[-] Lỗi: Không tìm thấy thư mục nguồn {source_dir}")
        return
        
    # Tạo các thư mục đích nếu chưa tồn tại
    for class_name in MALWARE_CLASSES.values():
        class_dir = os.path.join(target_base_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            
    all_files = os.listdir(source_dir)
    total_files = len(all_files)
    print(f"[+] Tìm thấy {total_files} tệp trong thư mục gốc. Bắt đầu xử lý...")
    
    counts = {class_name: len(os.listdir(os.path.join(target_base_dir, class_name))) for class_name in MALWARE_CLASSES.values()}
    deleted_count = 0
    moved_count = 0
    
    for idx, file_name in enumerate(all_files):
        file_path = os.path.join(source_dir, file_name)
        
        # Chỉ xử lý tệp tin
        if not os.path.isfile(file_path):
            continue
            
        # 1. Xác định nhóm mã độc trước
        class_name = get_malware_class(file_name)
        if class_name is None:
            continue
        
        # 2. Kiểm tra nếu nhóm này đã đủ 4000 tệp thì bỏ qua luôn, không sao chép thêm nữa
        if counts[class_name] >= limit_per_class:
            continue
            
        # 3. Kiểm tra cấu trúc PE thực thi hợp lệ
        if not is_valid_pe_with_code(file_path):
            # Tệp không phải PE Windows hợp lệ, bỏ qua để giữ an toàn thư mục gốc
            continue
            
        target_dir = os.path.join(target_base_dir, class_name)
        target_file_path = os.path.join(target_dir, file_name)
        
        # 4. Sao chép tệp tin vào thư mục đích
        try:
            if not os.path.exists(target_file_path):
                shutil.copy(file_path, target_file_path)
                counts[class_name] += 1
                moved_count += 1
        except Exception as e:
            print(f"[-] Lỗi khi sao chép {file_name}: {e}")
            
        if (idx + 1) % 500 == 0 or (idx + 1) == total_files:
            print(f"  -> Tiến trình: {idx + 1}/{total_files} tệp...")
            
    print("\n================ BÁO CÁO PHÂN LOẠI VÀ THU THẬP MÃ ĐỘC ================")
    print(f" - Tổng số tệp nguồn đã quét: {total_files}")
    print(f" - Số tệp đã sao chép thêm mới: {moved_count}")
    print(" - Chi tiết số lượng mẫu hiện tại trong mỗi thư mục loại mã độc:")
    for class_name, count in counts.items():
        print(f"   * {class_name.upper()}: {count}/{limit_per_class} tệp")
    print("======================================================================")

if __name__ == "__main__":
    SOURCE_FOLDER = "D:\malware\Virusshare.00470\Virusshare.00470"
    process_and_classify_dataset(SOURCE_FOLDER, target_base_dir="data", limit_per_class=4000)