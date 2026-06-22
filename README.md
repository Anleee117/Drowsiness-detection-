# DROWSINESS DETECTION WITH MEDIAPIPE & OPENCV

Hệ thống cảnh báo buồn ngủ thời gian thực (Real-time Drowsiness Detection) sử dụng **MediaPipe Face Landmarker** và **OpenCV** để phân tích trạng thái khuôn mặt thông qua webcam. Dự án được thiết kế nhằm đưa ra cảnh báo chính xác, giảm thiểu tối đa báo động giả (false alarms) nhờ cơ chế phân cấp cảnh báo thông minh và bộ lọc chuyển động nháy mắt tự nhiên.

---

## **1. Thông tin tổng quan (Overview)**
* **Mục tiêu:** Phát hiện trạng thái mệt mỏi, lờ đờ, ngủ gật (microsleep), mất tập trung hoặc quay mặt đi nơi khác của người lái xe hoặc người vận hành máy móc theo thời gian thực.
* **Nguyên lý hoạt động:** Thu nhận luồng hình ảnh từ camera -> Xác định và định vị mạng lưới khuôn mặt 3D -> Trích xuất tọa độ các vùng đặc trưng (mắt, miệng, hướng đầu) -> Tính toán các chỉ số sinh học -> So sánh với ngưỡng (threshold) và thời gian duy trì (counter) -> Phân loại cấp độ trạng thái và phát cảnh báo âm thanh khi cần thiết.

---

## **2. Cơ sở để lấy dữ liệu (Data Collection & Features)**
Hệ thống sử dụng mô hình lưới khuôn mặt **MediaPipe Face Mesh** (tải qua API `FaceLandmarker` dạng tệp `face_landmarker.task` siêu nhẹ ~4MB) để xác định **468 điểm mốc (landmarks)** tọa độ chuẩn hóa $(x, y, z)$. Các chỉ số được trích xuất dựa trên các mốc sau:

* **Mắt trái & Mắt phải (Mỗi mắt 6 điểm):**
  * Mắt phải: `[33, 160, 158, 133, 153, 144]`
  * Mắt trái: `[362, 385, 387, 263, 373, 380]`
  * *Mục đích:* Vẽ viền mắt và tính toán độ mở mắt (EAR).
* **Môi trong (8 điểm):**
  * Tọa độ góc trái, phải, môi trên, môi dưới: `[78, 308, 82, 13, 312, 87, 14, 317]`
  * *Mục đích:* Đo độ mở miệng (MAR) để phát hiện hành động ngáp.
* **Định vị tư thế đầu (6 điểm chính):**
  * Điểm chóp mũi (`1`), cằm (`152`), góc mắt ngoài trái (`33`), góc mắt ngoài phải (`263`), khóe miệng trái (`61`), khóe miệng phải (`291`).
  * *Mục đích:* Làm dữ liệu đầu vào cho giải thuật ước lượng góc quay đầu 3D (Head Pose Estimation).

---

## **3. Logic tính toán & Thuật toán xử lý (Processing Logic)**

Hệ thống hoạt động dựa trên các phép toán hình học và logic sau:

### **a. Tỷ lệ mở mắt - EAR (Eye Aspect Ratio)**
Công thức tính EAR cho một mắt dựa trên khoảng cách giữa các điểm mốc đối xứng dọc chia cho khoảng cách ngang:
$$\text{EAR} = \frac{\|P_2 - P_6\| + \|P_3 - P_5\|}{2 \cdot \|P_1 - P_4\|}$$
*Trong đó $P_i$ là các điểm mốc 2D.*

* **Bộ lọc phân biệt nháy mắt & Ngủ gật ngắn (Microsleep) dựa trên thời gian thực:**
  * Nháy mắt tự nhiên chỉ diễn ra trong thời gian ngắn (~0.1s - 0.4s).
  * Ngưỡng nhắm mắt hoàn toàn (Severe): `EAR < 0.14`.
  * Bộ đếm thời gian bắt đầu `eye_severe_start_time` ghi nhận thời điểm bắt đầu nhắm mắt. Nếu mở mắt (`EAR >= 0.14`), mốc thời gian lập tức được **reset về None**.
  * Chỉ khi mắt nhắm liên tục từ **>= 0.67 giây** (không phụ thuộc vào FPS của camera hay tốc độ máy tính), hệ thống mới kích hoạt trạng thái **Microsleep** và phát còi báo động. Điều này giúp loại bỏ hoàn toàn báo động giả do nháy mắt sinh lý.

### **b. Tỷ lệ mở miệng - MAR (Mouth Aspect Ratio)**
Công thức tính MAR sử dụng độ mở dọc của môi trong chia cho chiều rộng môi:
$$\text{MAR} = \frac{\|P_{\text{upper\_left}} - P_{\text{lower\_left}}\| + \|P_{\text{upper\_center}} - P_{\text{lower\_center}}\| + \|P_{\text{upper\_right}} - P_{\text{lower\_right}}\|}{2 \cdot \|P_{\text{left\_corner}} - P_{\text{right\_corner}}\|}$$
* Ngưỡng ngáp nhẹ (Mild Yawn): `MAR > 0.55` duy trì liên tục trong **0.67 giây**.
* Ngưỡng ngáp lớn (Severe Yawn): `MAR > 0.75` duy trì liên tục trong **0.33 giây**.

### **c. Ước lượng tư thế đầu (Head Pose Estimation)**
* Hệ thống sử dụng thuật toán **solvePnP (Perspective-n-Point)** của OpenCV để ánh xạ 6 điểm mốc 2D trên khuôn mặt vào mô hình 3D chuẩn của khuôn mặt người.
* Kết quả ma trận quay được chuyển đổi thành các góc Euler:
  * **Pitch (Cúi/Ngẩng đầu):** Nếu đầu cúi thấp vượt quá ngưỡng `HEAD_PITCH_SEVERE_THRESH = -20°` liên tục trong **2.5 giây**, hệ thống cảnh báo ngủ gật sâu (gục đầu). Thời gian trễ này giúp tài xế không bị còi báo lỗi khi chỉ cúi đầu nhìn bảng táp-lô trong chốc lát.
  * **Roll (Nghiêng đầu):** Nếu đầu nghiêng lệch sang hai bên vượt ngưỡng `HEAD_ROLL_SEVERE_THRESH = 35°` liên tục trong **2.5 giây**.

### **d. Logic đưa ra cảnh báo (Alarm Decision Matrix)**
Hệ thống kết hợp các chỉ số theo bảng logic phân cấp như sau:

| Trạng thái chỉ số | Phối hợp điều kiện | Trạng thái hiển thị | Còi báo động (`alarm.wav`) |
| :--- | :--- | :---: | :---: |
| **Bình thường** | EAR, MAR, Pitch, Roll đều trong ngưỡng an toàn | **AWAKE** (Xanh) | Không kêu |
| **Nhẹ (Mild)** | Chỉ ngáp nhẹ hoặc cúi đầu nhẹ đơn lẻ (Mắt vẫn mở to) | **CAUTION** (Cam) | Không kêu (Tránh làm phiền) |
| **Hợp tác nhẹ** | Mắt lờ đờ (`EAR < 0.16`) + Có thêm 1 dấu hiệu nhẹ khác (ngáp/cúi đầu) | **DROWSY!** (Đỏ) | **Kích hoạt** |
| **Nghiêm trọng (Severe)** | Bất kỳ chỉ số nào đạt mức nghiêm trọng (Nhắm mắt lâu, ngáp lớn, gục đầu sâu, nghiêng đầu mạnh) | **DROWSY!** (Đỏ) | **Kích hoạt lập tức** |
| **Mất dấu mặt (No Face)** | Không nhận diện thấy khuôn mặt liên tục quá **1.67 giây** | **DROWSY!** (Đỏ) | **Kích hoạt** |

---

## **4. Công nghệ lập trình sử dụng (Technologies)**
* **Python:** Ngôn ngữ lập trình chính, tối ưu cho xử lý ảnh và AI.
* **MediaPipe Tasks API (>= 0.10.21):** Trích xuất lưới khuôn mặt (Face Mesh) hiệu năng cao, chạy trực tiếp trên CPU mà không cần GPU chuyên dụng.
* **OpenCV (opencv-python >= 4.7.0):** Thu nhận camera, xử lý ma trận ảnh, thực hiện thuật toán ước lượng tư thế `solvePnP` và vẽ giao diện hiển thị thông tin thời gian thực.
* **NumPy (>= 1.21.0):** Hỗ trợ tính toán khoảng cách Euclidean giữa các điểm mốc trên ma trận nhiều chiều một cách tối ưu.
* **Multithreading (Threading):** Khởi chạy luồng phụ riêng cho âm thanh cảnh báo (`playsound`), giúp còi kêu độc lập không gây nghẽn (freeze) luồng xử lý video chính.

---

## **Các tính năng nổi bật**
* **Không cần dlib:** Loại bỏ dlib nặng nề, không cần cài đặt C++ Build Tools hay CMake phức tạp.
* **Không cần file model cục bộ nặng:** MediaPipe tự động tải mô hình siêu nhẹ (`face_landmarker.task` ~4MB) trong lần chạy đầu tiên.
* **Theo dõi 468 điểm (Face Mesh):** Độ chính xác cao hơn rất nhiều so với 68 điểm của dlib.
* **Cảnh báo đa cấp độ (Mild / Severe):**
  * **Mức độ nghiêm trọng (Severe):** Mắt nhắm tịt, ngáp rất to, cúi gập đầu -> Kêu ngay lập tức.
  * **Mức độ nhẹ (Mild):** Mắt lờ đờ + (kèm theo ngáp hoặc cúi đầu nhẹ) -> Kết hợp lại sẽ kêu. Nếu chỉ ngáp/cúi đầu nhẹ mà mắt vẫn mở to thì chỉ hiện cảnh báo "CAUTION" trên màn hình chứ không kêu ồn ào.

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
