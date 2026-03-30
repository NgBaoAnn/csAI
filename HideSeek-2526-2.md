# CSC14003 | Thực hành Nhập môn Trí tuệ Nhân tạo
# Bài thực hành
# Đấu trường Trốn Tìm (Hide and Seek Arena)

## 1. Giới thiệu
Đồ án 1 – Đấu trường Trốn Tìm nhằm giúp sinh viên hiểu, cài đặt và tối ưu hóa các thuật toán tìm kiếm thông qua một đấu trường tự động mô phỏng trò chơi trốn tìm, nơi các đội sinh viên có thể thi đấu với nhau để xếp hạng.

Mỗi đội sinh viên sẽ phát triển cả hai tác nhân (agent) sau:
- **Tác nhân Trốn (Hide agent):** Trốn được càng lâu càng tốt.
- **Tác nhân Tìm (Seek agent):** Tìm và bắt được càng nhanh càng tốt.

Các tác nhân sẽ được tự động ghép cặp trong Đấu trường để tính toán tỷ lệ thắng và xếp hạng.

## 2. Đấu trường (Arena)
Đấu trường là một framework tự động, được lập trình bằng Python, đóng vai trò là môi trường thi đấu cho các tác nhân. Dựa trên các bài nộp của các đội, Đấu trường sẽ (1) tự động nạp (import) các tác nhân của mỗi đội; (2) chạy một chuỗi các trận đấu giữa các đội; và (3) xuất kết quả.

Mô hình của Đấu trường được minh họa trong Hình 1.

*Hình 1: Tổng quan về mô hình Đấu trường.*

**Thuật toán 1: Vòng lặp tương tác giữa Tác nhân – Môi trường**
```text
while episode chưa kết thúc do
    ap ← PacmanAgent.step()
    ag ← GhostAgent.step()
    Environment.step(ap, ag)
    Cập nhật vị trí các tác nhân
    Kiểm tra va chạm hoặc trạng thái kết thúc
    s ← quan sát trạng thái môi trường mới
end while
```

Ở mỗi bước trong Thuật toán 1, thông tin được cung cấp bao gồm:
- Một bản đồ kích thước 21 × 21 (trong đó `0` là ô có thể đi qua và `1` là ô không thể đi qua).
- Vị trí hiện tại của hai tác nhân.
- Số bước hiện tại.

**Lưu ý:** Cả hai tác nhân đều nhận được thông tin trạng thái giống nhau và đưa ra quyết định đồng thời.

Các điều kiện chiến thắng được định nghĩa như sau:
- **Tác nhân Tìm (Seek) chiến thắng:** nếu Tác nhân Tìm chạm vào Tác nhân Trốn (khoảng cách Manhattan < 2).
- **Tác nhân Trốn (Hide) chiến thắng:** nếu đạt đến số bước cho phép tối đa.

Vui lòng lưu ý rằng, framework được cung cấp cho bạn để tham khảo, việc đánh giá tác nhân của chính bạn so với các điều kiện trên có thể cần một số tinh chỉnh.

Mã nguồn sẽ được cung cấp cho sinh viên với cấu trúc thư mục như sau:
```text
Arena/
    src/ ......................................Mã nguồn framework (KHÔNG chỉnh sửa)
    submissions/ ..................................................Không gian làm việc của sinh viên
        example_student/
            agent.py .............................................Bản cài đặt tham khảo
        student_id/
            agent.py ..............................................Tác nhân của đội sinh viên
    STUDENT_GUIDE.md ...............................................Hướng dẫn cho sinh viên
    run_game.sh .........................................................Script chạy nhanh
```

**Lưu ý quan trọng:** Nếu cả hai tác nhân di chuyển với cùng một tốc độ, Tác nhân Tìm không bao giờ có thể thắng vì nó không thể tiếp xúc với Tác nhân Trốn. Để đảm bảo tính công bằng, Tác nhân Tìm di chuyển với tốc độ 2 ô mỗi bước khi đi theo đường thẳng nhưng không thể đi theo hình chữ L ngay lập tức trong một lượt.

Tại mỗi bước, tác nhân của đội sinh viên phải trả về một hành động theo định dạng sau:
```python
class AgentInterface(ABC):
    @abstractmethod
    def step(self, map_state, my_position, enemy_position, step_number):
        # Đối với Tác nhân Pacman: trả về một hướng di chuyển (Move) hoặc (Move, steps), trong đó:
        # - steps là một số nguyên từ 1 đến tốc độ đường thẳng tối đa được cấu hình.
        # Đối với Tác nhân Ghost: trả về một hướng di chuyển (UP, DOWN, LEFT, RIGHT, hoặc STAY).
        pass
```

Các thuật toán được đề xuất được tóm tắt trong Bảng 1.

**Bảng 1: Các thuật toán đề xuất cho Tác nhân Tìm và Trốn**

| Tác nhân Tìm (Seek Agent) | Tác nhân Trốn (Hide Agent) |
| --- | --- |
| – Tìm kiếm theo chiều rộng (BFS) <br> – Tìm kiếm theo chiều sâu (DFS) <br> – A* <br> – Tham lam (Greedy) <br> – v.v. | – BFS: tìm vị trí xa nhất <br> – Minimax/Expectimax <br> – Monte Carlo <br> – Trường thế năng (Potential Fields) <br> – v.v. |


## 3. Giải đấu (Tournament)
Mỗi đội sẽ tham gia vào cả hai vai trò, Trốn và Tìm, và sẽ thi đấu với các tác nhân của tất cả các đội khác. Dựa trên kết quả, tỷ lệ thắng được tính như sau:

- `win_rate_hide` = (Số trận thắng khi Trốn) / (Tổng số trận Trốn)
- `win_rate_seek` = (Số trận thắng khi Tìm) / (Tổng số trận Tìm)

Kết quả được ghi lại trong bảng ví dụ sau.

**Bảng 2: Kết quả các trận đấu giữa các đội (Trốn vs. Tìm)**

| Trốn \ Tìm | Đội 1 | Đội 2 | Đội 3 | Đội 4 |
| --- | --- | --- | --- | --- |
| **Đội 1** | – | 0 | 1 | 1 |
| **Đội 2** | 0 | – | 1 | 0 |
| **Đội 3** | 1 | 0 | – | 1 |
| **Đội 4** | 0 | 0 | 1 | – |

Ở đây, giá trị `0` biểu thị tác nhân Trốn (Hide) thắng, trong khi giá trị `1` biểu thị tác nhân Tìm (Seek) thắng.

Dựa trên Bảng 2, một ví dụ về tỷ lệ thắng của Đội 1 là như sau:
- Trong vai trò tác nhân Trốn, Đội 1 thắng 1 trên 3 trận, dẫn đến `win_rate_hide` = 33.3%.
- Trong vai trò tác nhân Tìm, Đội 1 thắng 2 trên 3 trận, dẫn đến `win_rate_seek` = 66.6%.

Lưu ý rằng, bên cạnh kết quả thắng/thua đo được ở trên, chúng tôi cũng ghi lại số bước trung bình cần thiết để kết thúc mỗi trận đấu của tác nhân của bạn. Đối với Pacman, bạn nên cố gắng giảm thiểu giá trị này trong khi với Ghost, bạn nên tối đa hóa giá trị này. Để phá vỡ thế hòa (tie-breaking) giữa các đội có tỷ lệ thắng tương đương nhau, chúng tôi đánh giá sự chênh lệch giữa số bước trung bình của Pacman và Ghost; đội nào có sự chênh lệch thấp hơn sẽ được xếp hạng cao hơn.

Tiêu chí chấm điểm cho mỗi đội được xác định như sau:

| Tiêu chí | Điểm |
| --- | --- |
| Mức độ hoàn thiện trong cài đặt thuật toán | 3 |
| Xếp hạng ở lần nộp bài đầu tiên (xem tiến độ ở Mục 4) | Lên đến 3 |
| Xếp hạng ở lần nộp bài tối ưu hóa (xem tiến độ ở Mục 4) | Lên đến 4 |

## 4. Nộp bài (Submission)
Mỗi đội phải nộp một tệp `.zip` duy nhất lên Moodle có chứa cấu trúc thư mục sau đây (trong đó `group_id` là mã số nhóm của bạn):

```text
group_id/
    agent.py
    etc.
```

Tác nhân được nộp được phép import các tệp khác nằm trong cùng thư mục nộp bài.

**Lưu ý:** Sẽ không chấp nhận các bài nộp trễ hạn dưới bất kỳ hình thức nào.

*(Tiến độ nộp bài chưa được liệt kê chi tiết trong văn bản - The submission timeline is specified as follows)*

## 5. Các lưu ý (Notices)
Vui lòng chú ý đến các thông báo sau:
- Đây là bài tập NHÓM. Mỗi nhóm có từ 3 - 4 thành viên. Các nhóm có số lượng thành viên ít hơn hoặc nhiều hơn giới hạn quy định cần được sự chấp thuận của giảng viên thực hành.
- Đồ án này có hai giai đoạn nộp bài. Chỉ những đội đã nộp bài trong giai đoạn đầu tiên mới được phép nộp bài trong giai đoạn thứ hai.
- Các đội sinh viên phải đảm bảo rằng tác nhân của họ trả về một hành động trong vòng tối đa **1 giây**. Bạn có thể sử dụng thư viện `time` có sẵn cùng với tất cả các thư viện có sẵn đi kèm với Python 3. Các thư viện ngoài (external libraries) có sẵn cho đội của bạn bao gồm: `numpy`, `pandas`, `scipy`, và `gurobi`.
- Các công cụ AI KHÔNG bị cấm; tuy nhiên, sinh viên nên sử dụng chúng một cách khôn ngoan. Giảng viên thực hành có quyền tiến hành phỏng vấn vấn đáp bổ sung để đánh giá kiến thức của họ về đồ án.
- Bất kỳ hình thức đạo văn, gian lận, hoặc hành vi sai trái nào cũng sẽ dẫn đến điểm 0 cho môn học này.

Hết.
