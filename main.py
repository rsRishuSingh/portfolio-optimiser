import os
import json
from typing import List, TypedDict, Annotated, Sequence

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from utility import append_to_response, remove_think, get_context
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Portfolio optimization imports
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3-32b")

llm = ChatGroq(model=MODEL_NAME)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# TOOL: Sharpe Ratio
@tool
def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    arr = np.array(returns, dtype=float)
    excess = arr - risk_free_rate
    if arr.size < 2 or np.std(excess, ddof=1) == 0:
        raise ValueError("Insufficient data or zero volatility for Sharpe Ratio.")
    return float(np.mean(excess) / np.std(excess, ddof=1))

# Get historical data
@tool
def get_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices for the given tickers and date range.
    Returns a DataFrame with each column named by ticker.
    """
    data_frames = []
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, progress=False)["Adj Close"].rename(ticker)
        data_frames.append(df)
    # Concatenate on date index
    result = pd.concat(data_frames, axis=1)
    return result

# -------------------------------------------------------------------
# AGENT NODE: Input Query
# -------------------------------------------------------------------
def input_query(state: AgentState) -> AgentState:
    state["messages"] = []
    user_input = input("🤖 Enter your portfolio details (tickers, date range, risk tolerance):\nUser: ")
    query = HumanMessage(content=user_input)
    append_to_response([{"input_query": query}], filename="agent_log.json")
    return {"messages": [query]}

# -------------------------------------------------------------------
# AGENT NODE: Fetch Data
# -------------------------------------------------------------------
def fetch_data_agent(state: AgentState) -> AgentState:
    # Extract tickers and dates from last human message
    last_msg = state["messages"][-1].content
    # Here you'd parse out tickers, start, end; for example assume JSON
    params = json.loads(last_msg)
    tickers = params.get("tickers", [])
    start = params.get("start")
    end = params.get("end")
    df = get_data(tickers, start, end)
    append_to_response([{"fetched_data_head": df.head().to_dict()}], filename="agent_log.json")
    # Store DataFrame in context (not displayed directly)
    return {"messages": [AIMessage(content="Data fetched successfully."), df]}

# -------------------------------------------------------------------
# AGENT NODE: Calculate Weights
# -------------------------------------------------------------------
def calculate_frontliner(state: AgentState) -> AgentState:
    # Assume df is stored as second message
    df = state["messages"][1]
    # Calculate returns and risk/returns
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)
    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe()
    clean_weights = ef.clean_weights()
    append_to_response([{"optimized_weights": clean_weights}], filename="agent_log.json")
    return {"messages": [AIMessage(content=json.dumps(clean_weights))]}

# -------------------------------------------------------------------
# AGENT NODE: Return Predictor
# -------------------------------------------------------------------
def return_predictor(state: AgentState) -> AgentState:
    # Uses earlier predicted returns (from a forecasting tool) in context
    forecast = state.get("forecasted_returns")
    if not forecast:
        raise ValueError("Forecasted returns not found in state.")
    # Reconstruct price series and output
    output = {"forecasted_returns": forecast}
    append_to_response([{"final_output": output}], filename="agent_log.json")
    return {"messages": [AIMessage(content=json.dumps(output))]}
