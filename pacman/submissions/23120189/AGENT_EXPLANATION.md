# Hướng dẫn chi tiết thuật toán Tác nhân Pacman (Seeker Agent)

Tài liệu này giải thích chi tiết logic cốt lõi và các thuật toán được áp dụng bên trong file `agent.py` của sinh viên (`23120189`). Tác nhân Seeker này được xây dựng dựa trên các lý thuyết AI mạnh mẽ bao gồm **Cây tìm kiếm đối kháng (Adversarial Search)**, **Mô hình lập luận không chắc chắn (Belief Tracking)**, và **Cắt tỉa tối ưu (Alpha-Beta Pruning)**.

---

## 1. Tổng quan kiến trúc (Overview)

Pacman Seeker được chia làm **2 chiến lược chính** (tùy thuộc vào việc Pacman có nhìn thấy Ghost hay không):
- **Chiến lược A (Khi nhìn thấy Ghost):** Kích hoạt hệ thống "Săn đuổi bằng Minimax", tính toán trước 6 nước đi (Depth-6) để lùa Ghost vào ngõ cụt.
- **Chiến lược B (Khi Ghost tàng hình trong Fog of War):** Kích hoạt hệ thống "Theo dõi dấu vết (Belief Set)", suy luận vòng vây và tuần tra các khu vực chưa đuợc khám phá.

Để đảm bảo AI chạy dưới 1 giây/lượt, ngay tại lượt đi đầu tiên (Turn 1), Agent sẽ **tính toán trước toàn bộ khoảng cách (Precomputation - APSP)** trên bản đồ.

---

## 2. Các kỹ thuật tối ưu hóa Hiệu suất (Performance Optimizations)

Bởi vì thuật toán Minimax duyệt hàng nghìn trường hợp trong chưa tới 1 giây, bộ não của tác nhân ưu tiên tính toán trước (Tiền xử lý) mọi thao tác lặp đi lặp lại:

1. **Ma trận Floyd-Warshall/BFS (APSP)**: `_precompute_apsp()` tính toán trước khoảng cách ngắn nhất giữa tất cả các cặp ô trên bản đồ. Thay vì chạy BFS mỗi lượt để tìm đường cản Ghost, Pacman chỉ cần tra bảng `self.dist_matrix[p_row, p_col, g_row, g_col]`.
2. **Theo dõi nước đi (`_ghost_moves` & `_pac_dests`)**: Danh sách tất cả các điểm mà Ghost và Pacman có thể lướt tới tại mọi ô được tính sẵn 1 lần và lưu vào cache.
3. **Move Ordering (Alpha-Beta Pruning)**: Trong cây tìm kiếm, luôn sắp xếp duyệt các bước "làm Pacman gần Ghost nhất" trước. Việc này giúp thuật toán Alpha-Beta nhanh chóng cắt bỏ (Prune) cực kỳ nhiều nhánh xấu, ép độ sâu thuật toán từ 4 plies lên được **6 plies**.

---

## 3. Chiến lược A: Săn Ghost bằng Minimax (Depth-6)

Khi tọa độ của Ghost lộ diện, thuật toán Minimax kích hoạt:

### Môi trường đi song song (Simultaneous Moves)
Trong hệ thống Hide & Seek này, cả Pacman và Ghost ra quyết định cùng 1 lúc. Hàm `_ab_max` và `_ab_min` được thiết kế để Pacman và Ghost giả lập bước đi cùng nhau. Pacman và Ghost đều giả định đối phương sẽ đi nước đi khôn ngoan nhất, sau đó mới kiểm tra việc "Bắt được Ghost" (Distance/Manhattan == 0) và tính điểm.

### Hàm Đánh giá (Evaluation Function)
Khi tìm kiếm đạt mức độ sâu tối đa (Depth 6), Pacman sử dụng hàm `_evaluate` để chấm điểm trạng thái bàn cờ hiện tại. Điểm **CÀNG THẤP CÀNG TỐT** (vì Pacman là Minimizer).

Hàm đánh giá gồm 4 trọng số:
- **Khoảng cách (APSP * 10)**: Mục tiêu tối thượng của Pacman là áp sát Ghost.
- **Tiền đạo chặn cửa (Exit Control)**: Điểm trừ khổng lồ nếu khoảng cách từ Pacman đến lối thoát hiểm của Ghost ≤ 1 bước.
- **Khóa góc (Trap/Mobility Penalty)**: Phạt Ghost nếu Ghost đi vào ô có ít hàng xóm (ngõ cụt hoặc đường ống dài). Ép Ghost vào ngõ cụt sẽ làm điểm Evaluator giảm kịch liệt, lôi kéo Pacman đẩy Ghost vào góc tường. 
- **Manhattan Adjacency (Áp sát)**: Cực kì ưu tiên các ô nằm "kề mép" Ghost (Cách 1 ô) để ép Ghost không còn đường chéo.

---

## 4. Chiến lược B: Truy tìm ngẫu nhiên (Fog of War)

Khi Ghost đi vào vùng sương mù, tác nhân Pacman bắt đầu quy trình làm "Thám tử":

### Belief Set (Tập lập luận)
Mọi ô có khả năng chứa Ghost đều được gọi là **Belief**. Qua từng bước:
- **Mở rộng (Expand):** Ghost đi được 1 ô, tập Belief sẽ nở rộng ra các ô hàng xóm liền kề.
- **Cắt tỉa (Prune):** Nếu Pacman soi đèn và thấy 1 ô trống rỗng, ta gỡ ô đó từ tập Belief. Điều này dần dần dồn Ghost vào 1 bộ đếm tọa độ cực chật hẹp.

### Adaptive Hunt (Đi tuần tự động)
- **Truy lùng tâm vây (Centroid):** Nếu khoanh vùng Ghost nhỏ hẹp (Dưới 40% map), Pacman đâm thẳng vào tâm của điểm bao vây (Centroid).
- **Tuần tra rìa sương mù (Patrol):** Nếu Ghost trốn quá kỹ, Pacman ưu tiên lang thang qua những ô giáp bóng tối (`fog_adjacent`) để dò tìm.
- **Ngăn chặn Ocsillation (Gật gù tại chỗ):** Hàm được trang bị list sorting tránh việc Pacman ngập ngừng đứng giữa 2 đường cong. 
- **Trí nhớ dài hạn (`visit_count`)**: Trọng số cực kì đặc biệt. Mỗi cell Pacman đi qua sẽ tăng điểm phạt (`visit_count += 1`), và mọi ô dần quên đi theo thời gian (`* 0.97` Exponential Decay). Nhờ hàm này, Pacman luôn khát khao thám hiểm các khu vực nó chưa từng vào để dọn sạch Fog of War mà không bao giờ bị kẹt lại 1 góc map.

---

## Tổng kết (Summary)

Đây là vòng lặp hoàn hảo cho Agent Seek này:
1. Bạn không thấy nó? **-> Dùng Belief Set và Trừ lùi lượt truy cập (Visit Count) đi thám hiểm sương mù.**
2. Bạn nhìn thấy nó chớp tắt? **-> Ép belief dồn vòng vây vào chân tường.**
3. Bạn nhìn thấy nó rõ ràng? **-> Bật Alpha-Beta Minimax độ sâu 6, lùa Ghost vào ngõ cụt và giành chiến thắng tuyệt đối.**
