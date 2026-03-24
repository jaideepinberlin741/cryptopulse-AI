from binance.client import Client
import pandas as pd
import os


# Resolve project root and output directory robustly
script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
output_dir = os.path.join(project_root, "data", "raw")

os.makedirs(output_dir, exist_ok=True)


# Initialise Binance client (uses public endpoints if no keys)
client = Client()


# We store low-TF for intrabar aggregation + high-TF for training
TIMEFRAMES = [
    "5m",   # intrabar / live aggregation
    "15m", "1h", "4h", "1d",  # main training / inference TFs
]


def fetch_klines(symbol: str, interval: str, start_str: str = "1 Jan, 2020") -> pd.DataFrame:
    """Fetch historical klines for a symbol/interval and return a clean OHLCV DataFrame."""
    print(f"Fetching {symbol} {interval}...")
    klines = client.get_historical_klines(symbol, interval, start_str)

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_vol",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    # Keep only OHLCV + open_time; cast numeric columns appropriately
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df


def main():
    symbol = "BTCUSDT"

    for tf in TIMEFRAMES:
        df = fetch_klines(symbol, tf, start_str="1 Jan, 2020")

        csv_path = os.path.join(output_dir, f"btc_{tf}_raw.csv")
        df.to_csv(csv_path, index=False)
        print(f"  {len(df):,} rows → {csv_path}")

    print("Complete multi-timeframe raw dataset ready!")


if __name__ == "__main__":
    main()