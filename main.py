#!/usr/bin/env python
import matplotlib.pyplot as plt
from pycoingecko import CoinGeckoAPI
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any
from datetime import datetime
import numpy as np

cg = CoinGeckoAPI()
mcp = FastMCP("Crypto MCP")

@mcp.tool(description="Get the current price of a cryptocurrency from CoinGecko")
async def get_current_price(coin_id: str, vs_currency: str = "usd") -> Dict[str, Any]:

    data = cg.get_price(ids=coin_id, vs_currencies=vs_currency)
    if coin_id not in data:
        return {"error": f"Coin id '{coin_id}' not found"}
    return {
        "coin": coin_id,
        "currency": vs_currency,
        "price": data[coin_id][vs_currency],
    }




#For today:

#Install these in the terminal: pip install datetime, pip install matplotlib



import matplotlib.pyplot as plt
from pycoingecko import CoinGeckoAPI
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any
from datetime import datetime
import numpy as np






@mcp.tool(description="Get the current date and time.")
async def get_current_datetime() -> Dict[str, Any]:
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S")
    }



@mcp.tool(description="Count words and characters in a string.")
async def analyze_text(text: str) -> Dict[str, Any]:
    return {
        "word_count": len(text.split()),
        "character_count": len(text),
        "unique_words": list(set(text.lower().split()))
    }



#pip install numpy

@mcp.tool(description="Generate a plot of a math function given as a string, e.g., 'x**2 + 3*x - 5'")
async def generate_math_plot(equation: str) -> Dict[str, Any]:
    x = np.linspace(-10, 10, 400) 
    try:
        y = eval(equation, {"x": x, "np": np, "__builtins__": {}})
    except Exception as e:
        return {"error": f"Invalid equation: {e}"}

    plt.figure()
    plt.plot(x, y, label=f"y = {equation}", color='blue')
    plt.title(f"Plot of y = {equation}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {"message": f"Successfully plotted y = {equation}"}

if __name__ == "__main__":
    print("Starting Crypto MCP Server (Basic)…")
    mcp.run()
