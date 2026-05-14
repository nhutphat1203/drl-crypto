# Deep Reinforcement Learning cho Giao dịch Tiền mã hóa (DRL-Crypto)

## Tổng quan dự án
Dự án này cung cấp một framework hỗ trợ quá trình huấn luyện và kiểm thử các tác tử (agents) giao dịch tiền mã hóa tự động dựa trên phương pháp Học tăng cường Sâu (Deep Reinforcement Learning - DRL). Hệ thống sử dụng thuật toán PPO (Proximal Policy Optimization) kết hợp với các kiến trúc mạng nơ-ron nhằm xử lý dữ liệu chuỗi thời gian (như GRU, LSTM) để phân tích diễn biến thị trường và đưa ra các quyết định phân bổ danh mục đầu tư.

Dự án được xây dựng dựa trên các thư viện tiêu chuẩn trong lĩnh vực như `PyTorch`, `Gymnasium` (cho thiết kế môi trường học) và `Stable-Baselines3` (đóng gói sẵn các thuật toán RL).

## Kiến trúc Hệ thống
Hệ thống được thiết kế theo cấu trúc module, phân tách rõ ràng giữa xử lý dữ liệu, môi trường mô phỏng và phần huấn luyện:

* **`environment/`**: Định nghĩa môi trường tương tác (`market.py`) kế thừa từ `gymnasium.Env`. Môi trường này mô phỏng các biến động của thị trường, bao gồm việc khớp lệnh, trượt giá (slippage), chi phí giao dịch (fee), và chuyển đổi trạng thái của danh mục đầu tư.
* **`finance/`**: Chứa logic tính toán tài chính cốt lõi (`account.py`, `ohlcv.py`). Theo dõi số dư, trạng thái vị thế đang mở (open positions), và tính toán sự thay đổi trong tổng giá trị tài sản ròng (NAV) để cung cấp tín hiệu phần thưởng (reward).
* **`data_manager/`**: Đảm nhiệm việc tải, phân tách và cung cấp dữ liệu theo từng chu kỳ (episode) một cách liên tục cho môi trường thông qua các lớp như `DataProvider` và `Episode`.
* **`trainer/`**: Đóng gói quá trình huấn luyện mô hình. Module này bao gồm bộ trích xuất đặc trưng tùy chỉnh (`custom_extractor.py`) ứng dụng CNN, GRU, hoặc LSTM để xử lý dữ liệu lịch sử. Tệp `trainer.py` thiết lập thuật toán PPO cùng các cơ chế kiểm soát tiến trình học như Early Stopping (dừng sớm nếu không cải thiện) và Checkpoint định kỳ.
* **`preprocess/` & `scaled_data/`**: Chứa mã nguồn để làm sạch, tính toán các chỉ báo kỹ thuật (Technical Indicators) và chuẩn hóa dữ liệu đầu vào.
* **`scripts/`**: Chứa các kịch bản thực thi trực tiếp, bao gồm:
    * `download_data.py`: Tải dữ liệu OHLCV lịch sử thông qua API sàn giao dịch.
    * `train.py`: Bắt đầu quá trình huấn luyện.
    * `test.py` / `get_signal.py`: Đánh giá hiệu suất của mô hình sau huấn luyện và cung cấp tín hiệu giao dịch.
    * `plot_tensorboard.py`: Hỗ trợ biểu diễn trực quan các logs huấn luyện.
* **`backtest/`**: Tập hợp các công cụ hỗ trợ mô phỏng giao dịch lịch sử, nhằm đánh giá một cách độc lập độ tin cậy của chiến lược.

## Đặc tả Kỹ thuật
* **Không gian Trạng thái (Observation Space)**: Hệ thống sử dụng không gian hỗn hợp dạng từ điển (`Dict space`) bao gồm hai thành phần chính:
  1. `time_series`: Ma trận hai chiều biểu diễn chuỗi dữ liệu trong quá khứ (cửa sổ thời gian - window size). Nó chứa dữ liệu giá và khối lượng (OHLCV) cùng nhiều chỉ báo kỹ thuật đã được làm phẳng và chuẩn hóa.
  2. `portfolio_features`: Vector một chiều mô tả trạng thái hiện hành của danh mục (tỷ trọng tiền mặt, giá trị tài sản nắm giữ).
* **Không gian Hành động (Action Space)**: Hoạt động dựa trên không gian liên tục (thường mang dải giá trị từ `[0, 1]` hoặc `[-1, 1]`) biểu diễn cường độ và quyết định giao dịch (Ví dụ: 0 là bán, 1 là mua toàn bộ số dư).
* **Kiểm soát thông số học**: Được cấu hình bằng `config.yaml` hỗ trợ kiểm soát chặt chẽ các siêu tham số mạng như `learning_rate` (với tùy chọn giảm dần/linear schedule), `gamma`, số epochs, hệ số entropy (`ent_coef`), và tần suất đánh giá mạng (`eval_freq`).

## Cài đặt Môi trường
Dự án yêu cầu cài đặt Python 3.12.
Quy trình khởi tạo môi trường bao gồm thao tác cài đặt qua `requirements.txt`:
```bash
pip install -r requirements.txt
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Các nhóm thư viện cốt lõi bao gồm:
* **Học sâu và RL**: `torch>=2.0.0`, `gymnasium==1.2.3`, `stable_baselines3==2.7.1`, `sb3_contrib`
* **Xử lý số liệu**: `numpy`, `pandas`, `scikit-learn`
* **Phân tích giao dịch**: `ccxt`, `TA-Lib`, `vectorbt`

## Hướng dẫn Chạy Thử nghiệm (Workflow)
Các bước chạy được đóng gói trong scripts/
1. Tùy chỉnh các tham số cần thiết trong tệp `config.yaml`.
2. Tải dữ liệu bằng `scripts/download_data.py`
3. Tiền xử lý data bằng `scripts/preprocess_*.py`
4. Chuẩn hóa bằng `scripts/normalized_data.py`
5. Huấn luyện agent bằng `scripts/train.py`
6. Sử dụng agent để giao dịch và lấy dữ liệu môi trường bằng `scripts/get_signal.py`
7. Backtest bằng `scripts/test.py`
