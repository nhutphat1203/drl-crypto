import vectorbt as vbt
import pandas as pd

def evaluate_model_vs_benchmark(model_returns: pd.Series, benchmark_returns: pd.Series, 
                                model_name: str = "Agent", benchmark_name: str= "Benchmark",
                                init_cash: float = 1.0):
    # 1. Đồng bộ và xử lý dữ liệu
    model_returns = model_returns.asfreq('1h').fillna(0) if model_returns.index.freq is None else model_returns
    benchmark_returns = benchmark_returns.asfreq('1h').fillna(0) if benchmark_returns.index.freq is None else benchmark_returns

    # 2. Tính toán Stats
    annual_rf = 0.052
    hourly_rf = (1 + annual_rf)**(1/8760) - 1
    stats = model_returns.vbt.returns(freq='1h').stats(
        settings=dict(benchmark_rets=benchmark_returns, risk_free=hourly_rf, freq="1h")
    )

    # 3. Tính Equity (Tài sản)
    df_plot = pd.DataFrame({model_name: model_returns, benchmark_name: benchmark_returns})
    cum_equity = (1 + df_plot).cumprod() * init_cash

    # 4. Tạo Subplots: 1 hàng, 2 cột
    fig = vbt.make_subplots(rows=1, cols=2, 
                            shared_xaxes=False, 
                            horizontal_spacing=0.12,
                            subplot_titles=(f'So sánh Hiệu suất', f'Đường cong Equity'))

    # --- CỘT 1: So sánh (Dùng màu xanh và cam chuẩn Quant) ---
    cum_equity[model_name].vbt.plot(fig=fig, add_trace_kwargs=dict(row=1, col=1), 
                                    trace_kwargs=dict(name=model_name, line=dict(color='#1f77b4', width=2)))
    cum_equity[benchmark_name].vbt.plot(fig=fig, add_trace_kwargs=dict(row=1, col=1), 
                                       trace_kwargs=dict(name=benchmark_name, line=dict(color='#ff7f0e', width=2)))

    # --- CỘT 2: Equity Curve với Area Fill ---
    # fill='tozeroy' giúp tô màu vùng dưới đường line
    cum_equity[model_name].vbt.plot(fig=fig, add_trace_kwargs=dict(row=1, col=2), 
                                    trace_kwargs=dict(name=f"Equity {model_name}", 
                                                      line=dict(color='purple', width=2.5),
                                                      fill='tozeroy', 
                                                      fillcolor='rgba(128, 0, 128, 0.15)', # Màu tím nhạt trong suốt
                                                      showlegend=False))

    # 5. Tinh chỉnh Layout gọn gàng
    fig.update_layout(
        title_text=f"Lợi nhuận tích lũy",
        title_font=dict(size=18),
        height=400,  # Chiều cao vừa phải để bỏ vào tài liệu không bị thô
        width=1100, 
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.2), # Đẩy legend lên trên
        margin=dict(l=60, r=40, t=100, b=60)
    )

    # Cập nhật trục
    fig.update_yaxes(title_text="Giá trị tài sản", row=1, col=1)
    fig.update_yaxes(title_text="Giá trị tài sản", row=1, col=2)
    fig.update_xaxes(title_text="Thời gian", row=1, col=1)
    fig.update_xaxes(title_text="Thời gian", row=1, col=2)
    return stats, fig
