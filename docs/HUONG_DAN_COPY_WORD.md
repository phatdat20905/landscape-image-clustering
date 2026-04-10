## HƯỚNG DẪN COPY VÀO WORD

### Cấu trúc Chương 3 mới:

**3.1 GIỚI THIỆU**
- Mục tiêu chương
- Tổng quan workflow
- Lợi ích

**3.2 BƯỚC 1: LÀM SẠCH DỮ LIỆU (DATA CLEANING)**
- Mục tiêu
- Kỹ thuật xử lý phổ biến (bảng 5 kỹ thuật)
- Công cụ/Thư viện
- Kết quả thực tế (bảng thống kê)

**3.3 BƯỚC 2: TÍCH HỢP DỮ LIỆU (DATA INTEGRATION)**
- Mục tiêu
- Kỹ thuật chính
- Các bước xử lý (4 bước chi tiết)
- Kết quả Step 2 (thống kê + schema)

**3.4 BƯỚC 3: CHUYỂN ĐỔI DỮ LIỆU (DATA TRANSFORMATION)** [PHẦN LỚN NHẤT]
- Mục tiêu
- 3 kỹ thuật chính:
  1. Chuẩn hóa kích thước (Normalization - Pixel) + công thức toán
  2. Tăng cường dữ liệu (Augmentation) + bảng kỹ thuật + per-object JSON example
  3. Resize & định dạng (PNG 640×640)
- Kết quả Step 3 (lưu trữ + idempotency)

**3.5 BƯỚC 4: MÃ HOÁ DỮ LIỆU (ENCODING)**
- Mục tiêu
- 3 kỹ thuật trích xuất:
  1. CNN (ResNet50)
  2. HOG (Histogram of Oriented Gradients)
  3. Color Histogram
- Kết quả Step 4 (thống kê vectors + storage architecture)
- Lưu trữ Vector & Metadata (MinIO vs MongoDB breakdown)

**3.6 TỔNG HỢP PIPELINE**
- Sơ đồ dòng chảy (ASCII art + thống kê)
- Bảng tổng thể

**3.7 PHÂN TÍCH CHI TIẾT TỪNG LABEL**
- Bảng phân bố theo keyword (8 label)
- Phân bố sau augmentation

**3.8 SỰ IDEMPOTENCY VÀ RESUMABILITY**
- Thách thức
- Giải pháp từng step

**3.9 TINH CHỈNH & TỐI ƯU HÓA**
- Tinh chỉnh Normalization
- Tinh chỉnh Feature Extraction
- Tối ưu hóa lưu trữ

**3.10 KÊNH DỰ TOÁN & GHI CHÚ**
- Dung lượng (breakdown)
- Thời gian xử lý (bảng ước tính)
- Ghi chú & Caveats (4 điểm)

**3.11 KẾT LUẬN**
- Tóm tắt 4 bước
- Điểm nổi bật
- Kết quả cuối cùng

---

## CÁCH IMPORT VÀO WORD

### Option 1: Copy từ file Markdown
1. Mở file `Chuong3_Preprocessing_Moi.md` (đã lưu)
2. Đọc nội dung markdown
3. Copy từng phần vào Word
4. Format lại các heading, bảng, công thức (nếu cần)

### Option 2: Convert Markdown → Word (Tự động)
```bash
# Nếu có pandoc installed:
pandoc Chuong3_Preprocessing_Moi.md -o Chuong3_Preprocessing.docx

# Hoặc dùng online converter: https://pandoc.org/try/
```

### Option 3: Tôi viết thêm file Word template
- Tôi có thể tạo file `.docx` trực tiếp nếu bạn muốn (nhưng cần dùng python-docx library)

---

## CÁC ĐẶC ĐIỂM NỔI BẬT CÓ TRONG FILE MỚI

✅ **Kết hợp ảnh cũ:**
- Toàn bộ nội dung từ 4 ảnh PDF bạn gửi đã được tích hợp
- Thêm chi tiết code + công thức toán + thống kê thực tế

✅ **Mục tiêu rõ ràng:**
- Mỗi section bắt đầu với "Mục tiêu:" để rõ tại sao làm
- Input/Output cụ thể

✅ **Thống kê thực tế:**
- Từ EDA notebook: 12,096 → 8,033 → 11,329 → 4,451
- Bảng dung lượng, thời gian xử lý ước tính
- Per-label breakdown

✅ **Biểu đồ & Bảng:**
- ASCII art sơ đồ pipeline
- 8+ bảng thống kê/so sánh
- Công thức toán học (normalization, z-score)

✅ **Code examples:**
- Python snippets cho augmentation
- JSON per-object structure
- MongoDB index creation

✅ **Idempotency & Resumability:**
- Chi tiết cách Step 3, 4 hỗ trợ resume từ giữa
- DuplicateKeyError handling

✅ **Storage Architecture (NEW):**
- MinIO vectors (.npz) vs MongoDB metadata
- So sánh dung lượng: ~6.7 MB (MinIO) vs ~1.5 GB (nếu lưu JSON)

---

## NEXT STEPS

1. ✅ Đọc file `Chuong3_Preprocessing_Moi.md` 
2. ⚠️ Điều chỉnh / thêm thông tin cần thiết (tên giáo viên, trường, v.v.)
3. 📋 Copy vào Word với format đẹp
4. 🎨 Thêm hình ảnh EDA (từ `reports/preprocessing/*.png`)
5. ✔️ Review + submit

---

Bạn cần tôi giúp gì tiếp? 
- Sửa nội dung nào không hợp?
- Cần thêm section nào?
- Copy vào Word luôn?
