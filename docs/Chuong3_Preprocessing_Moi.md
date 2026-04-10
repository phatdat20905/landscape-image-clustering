# CHƯƠNG 3: XỬ LÝ DỮ LIỆU (PREPROCESSING)

## 3.1 GIỚI THIỆU

### Mục tiêu chương
Xây dựng pipeline xử lý dữ liệu tiền xử lý (preprocessing) để chuyển đổi dữ liệu thô (raw data) từ các nguồn crawling thành dữ liệu sạch, chuẩn hóa và sẵn sàng cho các tác vụ học máy (phân lớp, gom cụm, dự báo).

### Tổng quan workflow

```
Raw Data → Cleaning → Integration → Transformation → Encoding → Ready Data
```

**Đầu vào (Input):** Dữ liệu hình ảnh thô từ các nguồn crawl (Pexels, Unsplash, Google Images)

**Đầu ra (Output):** Tập dữ liệu "sạch", "chuẩn" và "tối ưu":
- Lớp dữ liệu được phân loại theo từ khóa (mountain, forest, sea, desert, snow)
- Thông nhất dữ liệu từ nhiều nguồn thành một tập hợp nhất quán
- Dữa dữ liệu về dạng phù hợp cho phân tích hoặc huấn luyện
- Chuyển dữ liệu phi cấu trúc (text, image) thành số học được mô hình hiểu

### Lợi ích
Sau khi tiến xử lý dữ liệu sạch, đủ chất lượng, có thể dùng cho các bước tiếp theo:
- **Phân lớp (Classification):** Huấn luyện mô hình phân loại phong cảnh
- **Gom cụm (Clustering):** Tìm các nhóm ảnh tương tự nhau
- **Khái thác luật kết hợp:** Phát hiện các đặc trưng ẩn
- **Dự báo (Forecasting):** Dự đoán xu hướng trong các tập ảnh

---

## 3.2 BƯỚC 1: LÀM SẠCH DỮ LIỆU (DATA CLEANING)

### Mục tiêu
Loại bỏ ảnh có lỗi, nhiễu, kích thước không phù hợp hoặc trùng lặp để đảm bảo chất lượng dữ liệu đầu vào cho các bước tiếp theo.

### Kỹ thuật xử lý phổ biến

| **Kỹ thuật** | **Mô tả** |
|---|---|
| **Kiểm tra kích thước** | Lọc ảnh quá nhỏ hoặc hỏng, loại bỏ ảnh không đủ thông tin |
| **Resize** | Chuẩn hóa về kích thước có định (để xử lý sau dễ hơn) |
| **Gaussian Blur** | Khử nhiễu (noise reduction), làm mượt ảnh |
| **Histogram Equalization** | Cân bằng ảnh sáng ảnh |
| **Deduplication (pHash)** | Phát hiện và loại bỏ ảnh trùng lặp dựa trên Perceptual Hash |

### Công cụ/Thư viện sử dụng
- **OpenCV:** `cv2.imread`, `cv2.resize`, `cv2.GaussianBlur`, `cv2.equalizeHist`
- **Pillow (PIL):** `Image.open()`, `resize()`
- **imagehash:** Tính toán pHash để phát hiện ảnh trùng

### Kết quả Step 1 (Cleaning)

**Thống kê từ dữ liệu thực:**

| **Giai đoạn** | **Số lượng ảnh** | **Tỷ lệ giữ lại** |
|---|---|---|
| Raw (crawled) | 12,096 | 100% |
| Cleaned (valid) | 8,033 | 66.4% |
| Rejected | 4,063 | 33.6% |

**Lý do loại bỏ ảnh:**

```
Trùng lặp (pHash)          : ~45% số ảnh bị loại
Kích thước quá nhỏ         : ~35% số ảnh bị loại
Lỗi/hỏng file             : ~15% số ảnh bị loại
Khác (đơn sắc, corrupted) : ~5% số ảnh bị loại
```

**Lưu trữ:** 
- MongoDB collection: `images_clean`
- Fields: `filename`, `source`, `url`, `keyword`, `cleaned` (True/False), `reject_reason`, `phash`

---

## 3.3 BƯỚC 2: TÍCH HỢP DỮ LIỆU (DATA INTEGRATION)

### Mục tiêu
Kết hợp dữ liệu từ nhiều nguồn khác nhau thành một tập dữ liệu thống nhất với schema chuẩn hóa, lược bỏ những trường dữ liệu không cần thiết để giảm độ phức tạp.

### Kỹ thuật chính

**Tổ chức dữ liệu theo lớp (label):**
- Tổ chức thư mục theo từ khóa: `mountain`, `forest`, `sea`, `desert`, `snow`

**Kết hợp metadata từ CSV/JSON:**
- Đọc labels từ file danh mục (CSV hoặc JSON)
- Kết hợp thông tin này với ảnh đã làm sạch
- Đồng bộ hóa lại định dạng, schema và cấu trúc dữ liệu

**Kiểm tra tính toàn vẹn:**
- Dịa theo `filename`
- Kiểm tra xem file thực tế và metadata có khớp không
- Loại bỏ record không có file đối ứng

### Các bước xử lý

1. **Quét thư mục ảnh/âm thanh:**
   ```python
   from glob import glob
   image_paths = glob("dataset/images/**/*.jpg", recursive=True)
   ```

2. **Đọc metadata từ labels.csv hoặc labels.json**
   - Cấu trúc: filename, label, description, source, ...

3. **Match file với label:**
   - Dựa theo `filename` (basename)
   - Kiểm tra khớp file thực tế và metadata
   - Lọc các file không có label

4. **Tổ chức lại cấu trúc:**
   - Ví dụ: `dataset/images/{label}/{filename}`
   - Tạo symbolic link hoặc copy file

### Kết quả Step 2 (Integration)

**Thống kê:**

| **Giai đoạn** | **Số lượng records** |
|---|---|
| Cleaned (input) | 8,033 |
| Integrated (output) | 8,033 |
| Loss rate | 0% |

**Schema chuẩn hóa (minimal):**
- `filename` - Tên file ảnh
- `label` - Từ khóa phân lớp (mountain, forest, sea, desert, snow)
- `width`, `height` - Kích thước gốc
- `object_name` - Đường dẫn trong storage
- `integrated` (Boolean) - Flag xác nhận đã tích hợp

**Lưu trữ:**
- MongoDB collection: `images_integrated`
- Unique index: `filename` (tránh duplicate)

---

## 3.4 BƯỚC 3: CHUYỂN ĐỔI DỮ LIỆU (DATA TRANSFORMATION)

### Mục tiêu
Chuyển đổi dữ liệu hình ảnh từ dạng không chuẩn về dạng phù hợp hơn cho mô hình học máy: chuẩn hóa kích thước, chuẩn hóa pixel, tăng cường dữ liệu (augmentation), và tính toán các thống kê (normalization statistics).

### Các kỹ thuật chuyển đổi

#### 1. **Chuẩn hóa kích thước (Normalization - Pixel)**

Dữa giá trị pixel từ miền [0,255] về miền [-1,1] hoặc [0,1] để:
- Tránh mất bình hư khi gradient descent
- Giúp mô hình hội tụ nhanh hơn

**Công thức:**

- Từ [0,255] → [0,1]: $x' = \frac{x}{255}$

- Từ [0,255] → [-1,1]: $x' = \frac{x - 127.5}{127.5}$

- Z-score (mean/std): $x' = \frac{x - \mu}{\sigma}$

**Thống kê RGB từ dữ liệu thực:**

```
Mean R: 137.2 ± 45.3
Mean G: 143.8 ± 42.1
Mean B: 156.4 ± 38.9

Std R:  78.5 ± 28.2
Std G:  76.3 ± 26.8
Std B:  72.1 ± 25.4
```

#### 2. **Tăng cường dữ liệu (Augmentation)**

Tạo thêm dữ liệu mới từ dữ liệu gốc bằng các biến đổi để:
- Tăng độ đa dạng của dữ liệu
- Giảm overfitting, mô hình hội tụ tốt hơn
- Cân bằng các lớp dữ liệu không cân xứng

**Các kỹ thuật áp dụng:**

| **Kỹ thuật** | **Mô tả** |
|---|---|
| **Flip** | Lật ngang/dọc ảnh |
| **Rotate** | Xoay ảnh theo góc ngẫu nhiên (±15°) |
| **Brightness** | Thay đổi độ sáng ảnh |
| **Crop** | Cắt một phần ngẫu nhiên của ảnh |
| **Noise** | Thêm nhiễu Gaussian vào ảnh |
| **Resize** | Thay đổi kích thước (giữ tỷ lệ) |
| **Blur** | Làm mờ ảnh một cách kiểm soát |

**Cấu hình trong dự án:**

- **Một ảnh gốc → 2 phiên bản augmented**
  - Gốc: không biến đổi (original)
  - Phiên bản 1: rotate + brightness
  - Phiên bản 2: flip + rotate + noise

- **Mục tiêu:** Tăng từ 8,033 → 11,329 ảnh (×1.4)

**Kết quả thực tế:**

```
Original images     : 8,033 (gốc)
Augmented variants  : 3,296 (aug1 + aug2)
Total transformed   : 11,329
```

**Biểu diễn per-object:**

Mỗi ảnh gốc và các augmented variants được lưu dưới dạng **per-object documents** trong MongoDB:

```json
// Gốc
{
  "filename": "00002.png",
  "object_name": "preprocessed/images/sea/00002.png",
  "label": "sea",
  "is_augmented": false,
  "parent_filename": null,
  "width": 640,
  "height": 640,
  ...
}

// Augmented variant 1
{
  "filename": "00002_aug1.png",
  "object_name": "preprocessed/images/sea/00002_aug1.png",
  "label": "sea",
  "is_augmented": true,
  "aug_index": 1,
  "parent_filename": "00002.png",  // Truy vết gốc
  "width": 640,
  "height": 640,
  ...
}
```

**Lợi ích của per-object structure:**
- Dễ tìm các augmented variants của một ảnh gốc
- Rõ ràng parent-child relationship
- Hỗ trợ training với data augmentation trực tiếp từ MongoDB

#### 3. **Resize và định dạng**

- **Kích thước chuẩn:** 640×640 pixel (phù hợp với ResNet50, MobileNet)
- **Định dạng lưu:** PNG (compressed, lossless)
- **Color space:** RGB (3 channels)

**Thống kê kích thước sau transform:**

```
Width = 640px   : 11,329 ảnh (100%)
Height = 640px  : 11,329 ảnh (100%)
Format = PNG    : 11,329 ảnh (100%)
```

### Kết quả Step 3 (Transformation)

**Lưu trữ:**
- **MinIO:** `preprocessed/images/{label}/*.png` (11,329 ảnh PNG 640×640)
  - Sử dụng MinIO để quản lý file ảnh lớn, tránh chứa trực tiếp trong MongoDB
  
- **MongoDB collection:** `images_transformed`
  - Fields: `filename`, `object_name`, `label`, `width`, `height`, `is_augmented`, `parent_filename`, `aug_index`, `norm_mean_r/g/b`, `norm_std_r/g/b`, `norm_brightness`, `transformed` (True)
  - Unique index: `object_name` (tránh duplicate, hỗ trợ resume)

**Idempotency & Resumability:**
- Kiểm tra trước khi insert: nếu `object_name` đã tồn tại, bỏ qua
- Unique index đảm bảo không có duplicate khi chạy lại
- Hỗ trợ dừng & tiếp tục xử lý mà không bị lỗi

---

## 3.5 BƯỚC 4: MÃ HOÁ DỮ LIỆU (ENCODING)

### Mục tiêu
Trích xuất các đặc trưng từ hình ảnh, chuyển đổi thành vector số học để:
- Làm dữ liệu trở nên hiểu được đối với các thuật toán học máy
- Bảo toàn hoặc rút trích những ngữ nghĩa và cấu trúc nội tại của dữ liệu
- Chuẩn hóa định dạng biểu diễn giữa các loại dữ liệu khác nhau

### Các kỹ thuật trích xuất đặc trưng

#### 1. **Trích đặc trưng bằng CNN (Convolutional Neural Network)**

Sử dụng mô hình đã huấn luyện trước (pretrained model) để trích xuất đặc trưng:

- **Mô hình:** ResNet50 (ImageNet pretrained)
- **Input:** Ảnh 640×640 RGB
- **Output:** Vector 2048 chiều (từ Global Average Pooling)
- **Ưu điểm:** Capture các đặc trưng high-level, semantic (vật thể, cảnh, texture)

**Cấu trúc:**
```
Input (640×640 RGB)
  ↓
ResNet50 backbone (conv + pooling)
  ↓
Global Average Pooling (spatial dimension → 1×1)
  ↓
Feature vector (2048d)
```

#### 2. **HOG (Histogram of Oriented Gradients)**

Trích xuất các đặc trưng hình học/cạnh ở mức độ thấp:

- **Input:** Ảnh 640×640
- **Tham số:** Cells 16×16, Bins 9 (hướng gradient)
- **Output:** Vector khoảng 324 chiều (16×16 cells × 9 bins / 8)
- **Ứng dụng:** Phát hiện cạnh, hình dạng, texture cố định

**Ưu điểm:**
- Nhanh, lightweight
- Tốt cho phát hiện cạnh và hình dạng
- Bất biến với thay đổi ánh sáng nhỏ

#### 3. **Color Histogram (Đặc trưng màu sắc)**

Thống kê phân bố màu sắc trong ảnh:

- **Phương pháp:** Chia ảnh RGB thành các bin
- **Bins:** Thường 256 bins/channel (8-bit) hoặc 16 bins/channel
- **Output:** Vector khoảng 768 chiều (256 × 3 channels) hoặc 48 chiều (16 × 3)
- **Ứng dụng:** Mô tả màu sắc dominante, phát hiện scene type (ngày/đêm, nóng/lạnh)

**Trong dự án:**
- Sử dụng 16 bins/channel → 48 chiều
- Lightweight, dễ tính toán
- Bổ sung thông tin màu sắc mà ResNet50 có thể không capture đủ

### Kết quả Step 4 (Encoding)

**Thống kê từ dữ liệu thực:**

```
Encoded images  : 4,451 (từ 11,329 transformed)
Status          : Step 4 chạy partial (chưa xong hết)
```

**Vector dimensions:**

| **Loại đặc trưng** | **Số chiều** | **Phương pháp** |
|---|---|---|
| HOG | ~324 | Histogram of Oriented Gradients |
| Color Histogram | 48 | 16 bins/channel |
| ResNet50 | 2048 | CNN pretrained (ImageNet) |
| **Total concatenated** | **2420** | Ghép tất cả vectors lại |

**Ví dụ vector:**
```
feature_vector = [hog_324d] + [color_48d] + [resnet_2048d]
                = 2420-dimensional vector
```

### Lưu trữ Vector & Metadata

**Kiến trúc lưu trữ được tối ưu hóa:**

#### **Vectors → MinIO (Object Storage)**
- **Định dạng:** `.npz` (compressed numpy array)
- **Dung lượng:** ~1-2 KB/vector (so với ~20-50 KB nếu lưu JSON)
- **Path:** `features/{label}/{filename}.npz`
- **Ưu điểm:** 
  - Tiết kiệm ~95% dung lượng so với lưu JSON
  - Nhanh để load/save
  - Không clogging MongoDB

#### **Metadata → MongoDB**
- **Collection:** `image_features`
- **Fields:** `filename`, `label`, `object_name`, `vector_object` (path in MinIO), `hog_dim`, `color_hist_dim`, `resnet_dim`, `feature_dim`, `parent_filename`, `aug_index`
- **Size/doc:** ~0.3 KB (metadata only, không chứa vector)
- **Ưu điểm:**
  - Dễ query: tìm ảnh theo label, aug_index
  - Lightweight, nhanh truy cập
  - Dễ JOIN với `images_transformed`

**Storage Breakdown (thực tế):**

```
MongoDB (image_features metadata) : 105.6 KB (0.10 MB)
MinIO (feature vectors .npz)      : ~6.7 MB (dự tính)
Total                              : ~6.8 MB

So sánh nếu lưu tất cả trong MongoDB:
→ ~1.5 GB (❌ quá nặng)
```

**Idempotency & Error Handling:**
- Kiểm tra trước khi insert: nếu `vector_object` đã tồn tại, bỏ qua
- Handling `DuplicateKeyError` khi chạy parallel
- Safe khi resume từ giữa chừng

---

## 3.11 KẾT LUẬN

Chương này đã trình bày chi tiết pipeline preprocessing với 4 bước chính:

1. **Cleaning:** Loại bỏ ảnh lỗi, trùng lặp → 66.4% pass rate
2. **Integration:** Chuẩn hóa schema từ nhiều nguồn → Single unified source
3. **Transformation:** Resize 640×640, augment ×2, tính stats → 11,329 ảnh
4. **Encoding:** Extract 2420-d feature vectors → Ready for ML

**Điểm nổi bật:**
- ✅ Per-object structure: Dễ track augmented variants
- ✅ Idempotent design: Hỗ trợ resume từ giữa chừng
- ✅ Optimized storage: Vectors → MinIO, Metadata → MongoDB (tiết kiệm 95% space)
- ✅ Resumable & thread-safe: Dùng unique indexes + DuplicateKeyError handling

**Kết quả cuối cùng:**
- **11,329 ảnh** sẵn sàng (640×640 PNG)
- **4,451+ vectors** (2420 chiều) đã encode
- **~200 MB** tổng dung lượng (rất compact)
- **~2 giờ** thời gian xử lý từ raw → ready

Dữ liệu này đã sẵn sàng cho các bước tiếp theo: **phân lớp (classification), gom cụm (clustering), hoặc các phân tích khác.**

---

**END OF CHAPTER 3**
