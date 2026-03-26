import pandas as pd
import quantstats as qs
import os

def get_data(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def backtest(path_strategy, path_benchmark, name):
    print(f"Bắt đầu backtest {name}")
    df = get_data(path_strategy)
    df_benchmark = get_data(path_benchmark)

    returns = df['equity'].pct_change().dropna()

    # Thiết lập tham số chu kỳ cho dữ liệu 1H Crypto
    # Thị trường truyền thống: 252 ngày x 1 nến ngày = 252
    # Thị trường Crypto 1H: 365 ngày x 24 giờ = 8760 nến/năm
    trading_periods_per_year = 365 * 24 

    benchmark_returns = df_benchmark['equity'].pct_change().dropna()

    returns.name = "PPO"
    benchmark_returns.name = "B&H"
    rf = 0.052

    folder_path = 'reports'
    os.makedirs(folder_path, exist_ok=True)


    # Xuất báo cáo HTML chuẩn Hedge Fund
    name_file = f'{name}_back_test_report.html'
    qs.reports.html(
        returns, 
        benchmark=benchmark_returns, 
        output=f'{folder_path}/{name_file}', 
        title='Báo cáo Backtest - Khung 1H', 
        periods_per_year=trading_periods_per_year,
        benchmark_title="B&H",
        rf=rf     
    )
    qs.plots.returns(
        returns, 
        benchmark=benchmark_returns, 
        savefig=f'{folder_path}/{name}_cumulative_returns.png',
        show=False,
    )

    cumulative_return = qs.stats.comp(returns)
    sharpe_ratio = qs.stats.sharpe(returns, rf=rf, periods=trading_periods_per_year)
    max_dd = qs.stats.max_drawdown(returns)
    
    benchmark_cumulative_return = qs.stats.comp(benchmark_returns)
    benchmark_sharpe_ratio = qs.stats.sharpe(benchmark_returns, rf=rf, periods=trading_periods_per_year)
    benchmark_max_dd = qs.stats.max_drawdown(benchmark_returns)

    print(f"Cumulative Return (Strategy vs Benchmark): {cumulative_return * 100:.2f}% - {benchmark_cumulative_return * 100:.2f}%")
    print(f"Sharpe Ratio (Strategy vs Benchmark):      {sharpe_ratio:.2f} - {benchmark_sharpe_ratio:.2f}")
    print(f"Max Drawdown (Strategy vs Benchmark):      {max_dd * 100:.2f}% - {benchmark_max_dd * 100:.2f}%")

    print(f"Đã xuất báo cáo thành công ra file {name_file}")

backtest(
    'signals/backtest_signal_ppo_btc.csv', 
    'signals/backtest_signal_buy_and_hold_btc.csv',
    'btc'
)

backtest(
    'signals/backtest_signal_ppo_eth.csv', 
    'signals/backtest_signal_buy_and_hold_eth.csv',
    'eth'
)

backtest(
    'signals/backtest_signal_ppo_bnb.csv', 
    'signals/backtest_signal_buy_and_hold_bnb.csv',
    'bnb'
)