# 🛡️ Advanced Malware & Ransomware Detection Engine Based on PE Header Graph Embedding & Deep Learning

Dự án này tái hiện, mở rộng và cải tiến phương pháp luận nghiên cứu từ bài báo khoa học: **"A novel approach for ransomware detection based on PE header using graph embedding"** (Journal of Computer Virology and Hacking Techniques, 2022). 

Hệ thống hoạt động dựa trên cơ chế **Phân tích tĩnh (Static Analysis)** cấu trúc phân bổ byte thô của PE Header (Windows Executable). Từ cấu trúc 1KB đầu tiên, hệ thống chuyển đổi thành đồ thị hướng có trọng số, áp dụng các kỹ thuật Máy học (Machine Learning), Học sâu (Deep Learning) và Học đồ thị (GNN) để phân loại mã độc đa lớp cũng như phát hiện bất thường.

---

## 🚀 Các Điểm Cải Tiến Cốt Lõi

1. **Phát hiện Bất thường với Graph Autoencoder (GAE):**
   * Sử dụng cấu trúc **GCN (Graph Convolutional Network)** tự định nghĩa trong [models.py](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/models.py) (Custom GCN Layer bằng PyTorch thuần để đảm bảo chạy ổn định trên mọi môi trường Windows/Linux mà không bị phụ thuộc phiên bản C++ compiler của `torch_geometric`).
   * Huấn luyện Graph Autoencoder (GAE) trên tập dữ liệu tệp sạch (Benign) nhằm học cách nén và tái thiết lập ma trận kề byte 2-gram bình thường. Tệp tin có lỗi tái tạo (MSE reconstruction loss) vượt quá ngưỡng chỉ định sẽ bị cảnh báo là bất thường (Mã độc).
2. **Phân loại Mã độc Đa lớp (Multi-class Classification):**
   * Thay vì chỉ phân loại nhị phân (Sạch vs Độc), hệ thống tự động trích xuất cấu trúc thư mục từ tập mẫu để phân chia chính xác thành 6 lớp trong [utils.py](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/utils.py):
     * **0 - Benign**: Tệp sạch hệ thống (EXE, DLL, MUI).
     * **1 - Backdoor**: Mã độc cửa sau.
     * **2 - Exploit**: Mã khai thác lỗ hổng bảo mật.
     * **3 - Worm**: Sâu máy tính.
     * **4 - Trojan**: Trojan ẩn mình độc hại.
     * **5 - Ransomware**: Mã độc tống tiền mã hóa.
3. **Khung Đối Chứng Đa Mô Hình (Multi-Model Evaluation Pipeline):**
   * Huấn luyện song song và đánh giá 8 mô hình khác nhau trực tiếp trên RAM thông qua [train_compare.py](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/train_compare.py):
     * *Học máy (ML)*: **Random Forest** (mô hình gốc của bài báo), **Decision Tree**, **SVM**, **XGBoost**.
     * *Học sâu (DL)*: **MLP** (Mạng nơ-ron đa tầng), **1D-CNN** (Mạng tích chập 1 chiều xử lý chuỗi byte).
     * *Học đồ thị (GNN)*: **GNN Classifier** (Mạng đồ thị có giám sát), **GAE Anomaly Detector** (Phát hiện bất thường bán giám sát).
   * Tạo bảng so sánh chi tiết các độ đo (Accuracy, Precision, Recall, Macro F1-score, Latency) và tự động vẽ biểu đồ so sánh `model/comparison_chart.png`.
4. **Engine Quét Tĩnh Đa Mô Hình:**
   * Module CLI Scanner trong [main.py](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/main.py) được nâng cấp để nạp đồng thời toàn bộ mô hình và in kết quả dự đoán đối chứng cạnh nhau cho người dùng kiểm thử trực quan.

---

## 📐 Kiến Trúc Luồng Dữ Liệu (Data Pipeline)

```text
[Tệp PE Thực thi] 
      │
      ▼ (extractor.py)
[Đọc 1024 Bytes PE Header]
      │
      ├──► (Dạng chuỗi byte chuẩn hóa) ──► [Mô hình 1D-CNN]
      │
      ▼
[Xây dựng đồ thị hướng 256 đỉnh bằng Byte 2-gram]
      │
      ├──► ( edge_index & edge_weight ) ──► [GNN Classifier / Graph Autoencoder]
      │
      ▼ (Thuật toán Power Iteration)
[Vector Nhúng Đặc Trưng 256 Chiều]
      │
      ▼
[Nạp và lưu trữ trực tiếp trên RAM]
      ├──► [train_compare.py] ──► Huấn luyện 8 mô hình đối chứng & Xuất biểu đồ
      └──► [main.py]          ──► Quét tĩnh đa mô hình thời gian thực (Static CLI Scanner)
```

---

## 📁 Cấu Trúc Thư Mục Dự Án

* [requirements.txt](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/requirements.txt): File định nghĩa môi trường thư viện phụ thuộc (được bổ sung `torch`, `xgboost`, `pandas`, `matplotlib`, `seaborn`, `tabulate`).
* [extractor.py](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/extractor.py): Xử lý trích xuất byte, xây dựng ma trận kề $256 \times 256$ và chạy Power Iteration.
* [utils.py](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/utils.py): Pipeline đọc và chuẩn bị dữ liệu đa định dạng (Vector, Chuỗi, Đồ thị GNN) kèm gán nhãn đa lớp.
* [models.py](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/models.py): Định nghĩa cấu trúc mạng nơ-ron PyTorch bao gồm [MLPClassifier](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/models.py#L5), [CNN1DClassifier](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/models.py#L22), [CustomGCNLayer](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/models.py#L52), [GraphAutoencoder](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/models.py#L97) và [GNNClassifier](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/models.py#L125).
* [collect.py](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/collect.py): Script tự động quét Hệ điều hành thu thập tập dữ liệu sạch cân bằng (EXE, DLL, MUI) về `data/benign/`.
* [move.py](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/move.py): Script lọc cấu trúc PE và di chuyển tệp mã độc về thư mục học máy.
* [train_compare.py](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/train_compare.py): Tập lệnh huấn luyện, đánh giá đối chứng toàn bộ 8 mô hình và xuất biểu đồ hiệu năng.
* [main.py](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/main.py): CLI Scanner tĩnh nạp toàn bộ các mô hình và quét tệp thời gian thực.

---

## 🛠️ Hướng Dẫn Cài Đặt Môi Trường

> [!CAUTION]
> **LƯU Ý BẢO MẬT AN TOÀN THÔNG TIN:** Toàn bộ quá trình thao tác giải nén mã độc thực tế và khởi chạy mã nguồn bắt buộc phải diễn ra trong môi trường Máy ảo cô lập hoàn toàn kết nối mạng (Host-Only hoặc Network Disconnected) để phòng tránh lây nhiễm hệ thống.

1. Khởi tạo môi trường ảo Python:
   ```bash
   python -m venv venv
   ```
2. Kích hoạt môi trường ảo:
   * Trên Windows (CMD): `venv\Scripts\activate.bat`
   * Trên Linux / macOS: `source venv/bin/activate`
3. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Quy Trình Vận Hành Hệ Thống

### Bước 1: Thu thập tập dữ liệu sạch (Benign)
Tự động quét lấy 1000 tệp EXE, 1000 tệp DLL và 1000 tệp MUI sạch từ Windows:
```bash
python collect.py
```

### Bước 2: Chuẩn bị tệp mã độc (Malware)
Trỏ biến `SOURCE_FOLDER` trong tệp [move.py](file:///d:/Thạc sĩ/IT6026 - AI trong ATTT/project/move.py) tới thư mục chứa mã độc thô tải về và chạy lọc cấu trúc PE Windows:
```bash
python move.py
```

### Bước 3: Chạy huấn luyện và so sánh hiệu năng các mô hình
Chạy quy trình huấn luyện đối chứng đồng thời 8 mô hình và trích xuất độ lỗi bất thường cho GAE:
```bash
python train_compare.py
```
Sau khi hoàn tất, bảng so sánh chi tiết hiệu năng sẽ được in ra Terminal và biểu đồ so sánh `model/comparison_chart.png` được lưu trữ.

### Bước 4: Chạy Engine quét đa mô hình thời gian thực
Khởi chạy ứng dụng Scanner tĩnh:
```bash
python main.py
```
Sao chép đường dẫn hoặc kéo thả trực tiếp một tệp thực thi bất kỳ vào màn hình terminal. Engine sẽ lập tức xử lý và đưa ra bảng dự đoán từ tất cả mô hình Máy học, Học sâu và GNN để đối chiếu trực quan.