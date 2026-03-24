Bạn có nhiệm vụ là 1 tester, hãy tạo 1 thư mục test và tạo các test case cho các file nghiệp vụ của tôi, nếu thiếu testcase file nào thì hãy thêm vào test case cho cập nhật đó và mỗi lần có file mới thì phải test lại hết, mọi thứ sẽ được chạy trong venv đã tạo sẵn. agent.md là file bộ não của bạn, bạn có quyền cập nhật nó để lưu trữ context 1 cách có hệ thống về dự án này.

## Context & Project Progress
- **Status:** Trainer & SB3 Pipeline Integration Tests [Done]
- **Architecture Updates:**
  - Validated YAML parsing into Dataclasses (`config.py`).
  - Added test coverage mimicking SB3 environments building strategies (`factory.py` with VecNormalize).
  - Mocked PPO and Callbacks to test `trainer.py` architectures reliably without GPU/CPU heavy simulation.

## Agent Logs/Notes
- **Testing Framework:** We use `pytest` for unit testing. Any AI Agent or Developer adding business logic files MUST include corresponding unit tests in the `test/` directory.
- **Run Tests:** To execute tests, run `venv\Scripts\pytest test/ -v` from the project root.
- **Gym Compatibility:** `environment/market.py` is fully compatible with Gymnasium APIs. Ensure any future agent interacts with `.step()` returning *(obs, reward, terminated, truncated, info)*.