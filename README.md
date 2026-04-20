# QC_FQL_Finetune
Finetune bài toán QC

1. Xử lý an toàn cho Logarit (Clipping)

Tuyệt đối không sử dụng trực tiếp log(D) vì có thể gây lỗi số học khi D → 0 hoặc D → 1.
Cách sửa:
Sử dụng công thức chuẩn:

R_disc =−log(1−D(s,a)+ϵ)

với:ϵ=1e −8 Hoặc công thức đối xứng: R_disc = log(D+ϵ)−log(1−D+ϵ)

👉 Giúp tránh log(0) và giữ reward trong vùng ổn định, không làm nổ Q-values.

2. Chuẩn hóa thay vì nhân hệ số cảm tính

Thay vì nhân trực tiếp D với hệ số như 5 hoặc 8, hãy chuẩn hóa tín hiệu từ discriminator.
Cách sửa: Theo dõi mean và std của R_disc, sau đó:

R_ disc_norm = (R_disc - mean) / (std+ϵ)
Sau đó mới nhân với hệ số \beta.

👉 Điều này giúp R_disc và R_env cùng scale, tránh mất cân bằng.

3. Lịch trình suy giảm (Annealing) cho β

Discriminator hữu ích ở giai đoạn đầu, nhưng về sau agent cần tập trung vào reward môi trường.

Cách sửa:

Khởi đầu: β = 0.1 hoặc 0.05
Giảm tuyến tính về 0 tại ~70% tổng số bước train

👉 Cuối training:
R_new ≈ R_env
	​
4. Nới lỏng ràng buộc ở pha Online
Với ~1 triệu bước online, agent có đủ thời gian khám phá.
Cách sửa:
Giảm hệ số \alpha: α = 10 → 5 → 1

👉 Giúp agent thoát khỏi bias của dữ liệu offline và tìm policy tốt hơn.
