import vectorbt as vbt
import pandas as pd

def evaluate_model_vs_benchmark(model_returns: pd.Series, benchmark_returns: pd.Series, 
                                model_name: str = "Agent", benchmark_name: str= "Benchmark"):
    # 1. Đồng bộ và xử lý dữ liệu
    model_returns = model_returns.asfreq('1h').fillna(0) if model_returns.index.freq is None else model_returns
    benchmark_returns = benchmark_returns.asfreq('1h').fillna(0) if benchmark_returns.index.freq is None else benchmark_returns

    # 2. Tính toán Stats
    annual_rf = 0.052
    hourly_rf = (1 + annual_rf)**(1/8760) - 1
    stats = model_returns.vbt.returns(freq='1h').stats(
        settings=dict(benchmark_rets=benchmark_returns, risk_free=hourly_rf, freq="1h")
    )

    # 3. Tính Lợi nhuận tích lũy (Chuyển sang phần trăm %)
    df_plot = pd.DataFrame({model_name: model_returns, benchmark_name: benchmark_returns})
    # Công thức: (Tích lũy lợi nhuận - 1) * 100 để ra %
    cum_returns_pct = ((1 + df_plot).cumprod() - 1) * 100

    # 4. Tạo Figure (1 hàng, 1 cột)
    fig = vbt.make_subplots(rows=1, cols=1)

    # --- Vẽ đường Agent và Benchmark ---
    cum_returns_pct[model_name].vbt.plot(fig=fig, 
                                         trace_kwargs=dict(name=model_name, line=dict(color='#1f77b4', width=2)))
    cum_returns_pct[benchmark_name].vbt.plot(fig=fig, 
                                             trace_kwargs=dict(name=benchmark_name, line=dict(color='#ff7f0e', width=2)))

    # 5. Tinh chỉnh Layout gọn gàng
    fig.update_layout(
        title_text="So sánh Lợi nhuận tích lũy (%)",
        title_font=dict(size=18),
        height=400,  
        width=900,   # Thu hẹp width lại một chút vì chỉ còn 1 biểu đồ
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5), # Căn giữa legend
        margin=dict(l=60, r=40, t=80, b=60),
        hovermode="x unified" # Hiển thị tooltip so sánh cả 2 đường cùng lúc khi rê chuột
    )

    # Cập nhật nhãn trục
    fig.update_yaxes(title_text="Lợi nhuận (%)")
    fig.update_xaxes(title_text="Thời gian")

    return stats, fig