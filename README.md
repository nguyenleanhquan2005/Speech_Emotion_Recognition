# Speech Emotion Recognition (SER)

Dự án nghiên cứu và triển khai mô hình nhận dạng cảm xúc từ giọng nói (Speech Emotion Recognition – SER) sử dụng các kiến trúc học sâu (Deep Learning). Hệ thống phân tích tín hiệu âm thanh ở mức vật lý (acoustic level) thay vì nội dung ngôn ngữ, tập trung vào các đặc trưng như **Fundamental Frequency (F0)**, **Formants (F1, F2, F3)** và các đặc trưng phổ để phân loại cảm xúc.

---

## 🎯 Mục tiêu

Mục tiêu của dự án là xây dựng và so sánh nhiều kiến trúc mạng học sâu trong bài toán phân loại cảm xúc từ tín hiệu giọng nói.

Bài toán được định nghĩa như sau:

P(emotion | audio_signal)

Trong đó:

- **Input**: file âm thanh `.wav`
- **Output**: nhãn cảm xúc (ví dụ: happy, sad, angry, neutral, fear…)

---

## 🧠 Cơ sở lý thuyết

Khác với phân tích văn bản, SER không dựa vào ngữ nghĩa câu nói mà dựa vào **đặc trưng âm học (acoustic features)** phản ánh trạng thái cảm xúc thông qua:

### 1️⃣ Fundamental Frequency (F0 – Pitch)

- Đại diện cho cao độ của giọng nói.
- Pitch cao và biến thiên mạnh thường liên quan đến *angry* hoặc *excited*.
- Pitch thấp và ít biến thiên thường liên quan đến *sad* hoặc *calm*.

---

### 2️⃣ Formants (F1, F2, F3)

- Là các đỉnh cộng hưởng của đường phát âm.
- Phản ánh cấu trúc phổ và đặc điểm âm sắc.
- Giúp mô hình phân biệt sự thay đổi trong cấu trúc âm thanh khi cảm xúc thay đổi.

---

### 3️⃣ Energy (Cường độ)

Energy = ∑ x(t)^2

- Năng lượng cao → cảm xúc mạnh (angry, excited).
- Năng lượng thấp → cảm xúc trầm (sad).

---

### 4️⃣ Đặc trưng phổ (Spectral Features)

- Mel-spectrogram
- MFCC (Mel-Frequency Cepstral Coefficients)

Các đặc trưng này biểu diễn âm thanh trong miền thời gian – tần số, giúp mạng CNN xử lý tương tự như ảnh.

---

## ⚙️ Pipeline xử lý

1. Load dữ liệu âm thanh (.wav)
2. Chia frame (20–40ms)
3. Trích xuất đặc trưng (F0, Formants, MFCC, Spectrogram)
4. Chuẩn hóa dữ liệu
5. Huấn luyện mô hình Deep Learning
6. Đánh giá bằng Accuracy, F1-score
7. Lưu trọng số mô hình (.pth)
8. Ghi log và trực quan hóa bằng Weights & Biases

---

## 🏗 Kiến trúc mô hình

Dự án thử nghiệm nhiều kiến trúc khác nhau để so sánh hiệu suất:

### 🔹 Shallow Neural Network

- Fully Connected layers
- Phù hợp với feature vector truyền thống (F0, Formants, MFCC)

---

### 🔹 Deep Neural Network (DNN)

- Nhiều tầng ẩn
- Học biểu diễn phi tuyến phức tạp hơn

---

### 🔹 Long Short-Term Memory (LSTM)

- Phù hợp với dữ liệu chuỗi thời gian
- Học được sự thay đổi cảm xúc theo thời gian

---

### 🔹 Residual Network (ResNet)

- Áp dụng trên Spectrogram
- Sử dụng skip connections để huấn luyện mạng sâu ổn định hơn

---

## 📊 Theo dõi huấn luyện

Quá trình huấn luyện được theo dõi bằng **Weights & Biases (WandB)**:

- Training Loss
- Validation Loss
- Accuracy
- Confusion Matrix
- Learning Curves

Giúp so sánh trực quan giữa các kiến trúc và tối ưu mô hình hiệu quả hơn.

---

## 🚀 Tính năng chính

- Phân tích đặc trưng âm thanh từ các file định dạng `.wav`.
- Thử nghiệm và so sánh hiệu suất trên nhiều kiến trúc mạng khác nhau:
  - **Deep Neural Network (DNN)**
  - **Long Short-Term Memory (LSTM)**
  - **Residual Network (ResNet)**
  - **Shallow Neural Network**
- Theo dõi quá trình huấn luyện trực quan qua **Weights & Biases (WandB)**.

---

## 📁 Cấu trúc thư mục


## 📁 Cấu trúc thư mục
```text
Asm/
├── Project_SER/           # Thư mục chứa mã nguồn chính
├── wav/                   # Dữ liệu âm thanh đầu vào (.wav)
├── wandb/                 # Nhật ký huấn luyện và biểu đồ (WandB)
├── speed_emotion.ipynb    # Notebook chính thực hiện huấn luyện và thử nghiệm
├── *.pth                  # Các trọng số (weights) của các mô hình tốt nhất
└── emotion_recognition_results.json # Kết quả dự đoán chi tiết  
