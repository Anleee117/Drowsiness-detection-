# DROWSINESS DETECTION WITH MEDIAPIPE & OPENCV

Hệ thống cảnh báo buồn ngủ sử dụng **MediaPipe Face Landmarker** và **OpenCV** để phân tích khuôn mặt theo thời gian thực. Hệ thống theo dõi đồng thời 4 yếu tố để đưa ra cảnh báo chính xác:

1. **EAR (Eye Aspect Ratio):** Nhắm mắt (Mắt là yếu tố chính - Primary Gate)
2. **MAR (Mouth Aspect Ratio):** Ngáp
3. **Head Pitch (Cúi đầu):** Ngủ gật gục đầu
4. **Head Roll (Nghiêng đầu):** Đầu nghiêng lệch sang hai bên
5. **Mất dấu khuôn mặt (No Face):** Quay mặt đi hướng khác quá lâu

---

## **Các tính năng nổi bật**
- **Không cần dlib:** Loại bỏ dlib nặng nề, không cần cài đặt C++ Build Tools hay CMake phức tạp.
- **Không cần file model cục bộ nặng:** MediaPipe tự động tải mô hình siêu nhẹ (`face_landmarker.task` ~4MB) trong lần chạy đầu tiên.
- **Theo dõi 468 điểm (Face Mesh):** Độ chính xác cao hơn rất nhiều so với 68 điểm của dlib.
- **Cảnh báo đa cấp độ (Mild / Severe):**
  - **Mức độ nghiêm trọng (Severe):** Mắt nhắm tịt, ngáp rất to, cúi gập đầu -> Kêu ngay lập tức.
  - **Mức độ nhẹ (Mild):** Mắt lờ đờ + (kèm theo ngáp hoặc cúi đầu nhẹ) -> Kết hợp lại sẽ kêu. Nếu chỉ ngáp/cúi đầu nhẹ mà mắt vẫn mở to thì chỉ hiện cảnh báo "CAUTION" trên màn hình chứ không kêu ồn ào.

---

## **Hướng dẫn cài đặt từ đầu (Cho máy tính hoàn toàn mới)**

Nếu bạn mang source code này sang một máy tính chưa từng cài đặt gì, hãy làm theo các bước sau:

### **Bước 1: Cài đặt Python**
1. Tải và cài đặt **Python** (phiên bản từ 3.8 đến 3.11 là tốt nhất) từ trang chủ: https://www.python.org/downloads/
2. **QUAN TRỌNG:** Trong quá trình cài đặt Python, nhớ tích vào ô **"Add Python to PATH"** (hoặc "Add python.exe to PATH") ở màn hình đầu tiên.

### **Bước 2: Tạo môi trường ảo (Virtual Environment - Khuyên dùng)**
Môi trường ảo giúp các thư viện của dự án này không bị xung đột với các dự án khác trên máy.
1. Mở Terminal (hoặc PowerShell, Command Prompt) tại thư mục chứa code (thư mục `Drowsiness-detection-with-OpenCV`).
2. Chạy lệnh tạo môi trường ảo (tên là `venv`):
   ```powershell
   python -m venv venv
   ```
3. Kích hoạt môi trường ảo:
   - Trên **Windows**:
     ```powershell
     .\venv\Scripts\activate
     ```
   - Trên **Mac/Linux**:
     ```bash
     source venv/bin/activate
     ```
   *(Sau khi kích hoạt, bạn sẽ thấy chữ `(venv)` hiện ở đầu dòng lệnh).*

### **Bước 3: Cài đặt thư viện**
Khi môi trường ảo đã được kích hoạt, bạn tiến hành cài các thư viện cần thiết bằng lệnh sau:
```powershell
pip install mediapipe>=0.10.21 opencv-python>=4.7.0 numpy>=1.21.0 playsound==1.2.2
```
*(Lưu ý: Chúng tôi sử dụng `playsound==1.2.2` vì phiên bản 1.3.0 trên Windows đôi khi gây lỗi).*

---

## **Cách chạy chương trình**

Đảm bảo bạn vẫn đang ở trong môi trường ảo `(venv)`, chạy lệnh sau:

```powershell
python detect_drowsiness_mine.py --alarm alarm.wav
```

**Thao tác trong lúc chạy:**
- Lần đầu chạy, hệ thống sẽ mất vài giây để tải file `face_landmarker.task` (~4MB) từ Google. Các lần sau sẽ chạy ngay lập tức mà không cần mạng.
- Để **thoát chương trình**, hãy click chuột vào cửa sổ camera (chỗ đang quay video) và nhấn phím **`q`** trên bàn phím.

---

## **Giao diện & Cảnh báo**
- Dưới cùng góc trái sẽ hiển thị các chỉ số **EAR, MAR, Pitch, Roll** theo thời gian thực (Xanh = Bình thường, Cam = Nhẹ, Đỏ = Nghiêm trọng).
- Trên cùng bên phải sẽ báo trạng thái **AWAKE** (Tỉnh táo), **CAUTION** (Chú ý), hoặc **DROWSY!** (Buồn ngủ).
- Hình vẽ trên mặt:
  - Mắt: Xanh lá cây
  - Miệng: Tím
  - Trục mũi (Đường thẳng màu xanh dương): Thể hiện hướng xoay của đầu (3D Pose).
