#!/usr/bin/env python
"""
Crypto MCP Server – ML model version

Two MCP tools:
  1. get_current_price – fetch current crypto price from CoinGecko
  2. predict_price     – fetch historical data and predict future prices using RandomForestRegressor

Requirements:
  pip install fastmcp pycoingecko pandas scikit-learn numpy
"""

import pandas as pd
import numpy as np
from pycoingecko import CoinGeckoAPI
from sklearn.ensemble import RandomForestRegressor
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any

# Initialise CoinGecko and MCP
cg = CoinGeckoAPI()
mcp = FastMCP("Crypto MCP Server (ML)")

@mcp.tool(description="Get the current price of a cryptocurrency from CoinGecko")
async def get_current_price(coin_id: str, vs_currency: str = "usd") -> Dict[str, Any]:
    """
    coin_id: e.g. 'bitcoin'
    vs_currency: e.g. 'usd'
    """
    data = cg.get_price(ids=coin_id, vs_currencies=vs_currency)
    if coin_id not in data:
        return {"error": f"Coin id '{coin_id}' not found"}
    return {
        "coin": coin_id,
        "currency": vs_currency,
        "price": data[coin_id][vs_currency],
    }

@mcp.tool(description="Fetch historical prices from CoinGecko and predict future prices using an ML model")
async def predict_price(coin_id: str, vs_currency: str = "usd",
                        days_history: int = 90, days_forecast: int = 7) -> Dict[str, Any]:
    """
    coin_id: e.g. 'bitcoin'
    vs_currency: e.g. 'usd'
    days_history: how many past days to use for training
    days_forecast: how many days ahead to forecast
    """
    # Fetch history
    raw = cg.get_coin_market_chart_by_id(id=coin_id,
                                         vs_currency=vs_currency,
                                         days=days_history)
    df = pd.DataFrame(raw['prices'], columns=['timestamp','price'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('date').sort_index()

    # Build lag features
    lags = 5
    for i in range(1, lags+1):
        df[f'lag_{i}'] = df['price'].shift(i)
    df = df.dropna()

    X = df[[f'lag_{i}' for i in range(1, lags+1)]]
    y = df['price']

    # Fit ML model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Use last known lags to predict forward recursively
    last_vals = list(df['price'].tail(lags).values[::-1])
    predictions = []
    for _ in range(days_forecast):
        x_pred = np.array(last_vals[:lags])[::-1].reshape(1, -1)
        next_price = model.predict(x_pred)[0]
        predictions.append(next_price)
        last_vals = [next_price] + last_vals[:-1]

    forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days_forecast)
    forecast_df = pd.DataFrame({'date': forecast_dates, 'predicted_price': predictions})

    return {
        "coin": coin_id,
        "currency": vs_currency,
        "history_days": days_history,
        "forecast_days": days_forecast,
        "predictions": forecast_df.to_dict(orient='records')
    }

if __name__ == "__main__":
    print("Starting Crypto MCP Server (ML)…")
    mcp.run()
