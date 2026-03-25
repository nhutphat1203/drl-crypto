Bạn có nhiệm vụ là 1 tester, hãy tạo 1 thư mục test và tạo các test case cho các file nghiệp vụ của tôi, nếu thiếu testcase file nào thì hãy thêm vào test case cho cập nhật đó và mỗi lần có file mới thì phải test lại hết, mọi thứ sẽ được chạy trong venv đã tạo sẵn. agent.md là file bộ não của bạn, bạn có quyền cập nhật nó để lưu trữ context 1 cách có hệ thống về dự án này.

## Context & Project Progress
- **Status:** Data Preprocessing & Script Suite Tests [Done]
- **Architecture Updates:**
  - Resolved `NameError` crash due to missing `import pandas`, `numpy`, and `talib` inside `dataprocess.py`.
  - Added robust preprocessing test coverage simulating feature engineering against rolling windows (`test_preprocess_data.py`).
  - Total Pytest coverage is fully covering the core RL, Config, and Preprocessing structures safely.

## Agent Logs/Notes
- **Testing Framework:** We use `pytest` for unit testing. Any AI Agent or Developer adding business logic files MUST include corresponding unit tests in the `test/` directory.
- **Run Tests:** To execute tests, run `venv\Scripts\pytest test/ -v` from the project root.
- **Gym Compatibility:** `environment/market.py` is fully compatible with Gymnasium APIs. Ensure any future agent interacts with `.step()` returning *(obs, reward, terminated, truncated, info)*.