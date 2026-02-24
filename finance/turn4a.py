#!/usr/bin/env python3
"""
Stock Market Simulation CLI App
================================
A fully offline mock stock market simulator with:
- 10 realistic mock stocks
- Random but realistic daily price movements
- CLI-based price charts (last 10 days)
- Portfolio management with buy/sell
- P&L tracking and analysis
- 50-day simulation with day skipping
- Restart and exit options

Uses only Python standard library — no external dependencies.
"""

import random
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
STARTING_BALANCE = 20_000.00
MAX_DAYS = 50
CHART_HEIGHT = 12
CHART_WIDTH = 10  # last 10 days
RISK_FREE_RATE = 0.05
TRADING_DAYS_PER_YEAR = 252

# ─────────────────────────────────────────────
# Stock Definitions
# ─────────────────────────────────────────────
STOCK_DEFINITIONS = [
    # (ticker, name, initial_price, volatility, drift, sector, industry)
    # volatility = daily std dev as fraction of price
    # drift = slight daily bias (positive = bullish)
    ("AAPL", "Apple Inc.", 182.50, 0.018, 0.0005, "Technology", "Consumer Electronics"),
    ("MSFT", "Microsoft Corp.", 378.90, 0.016, 0.0006, "Technology", "Software"),
    ("GOOG", "Alphabet Inc.", 141.20, 0.020, 0.0004, "Technology", "Internet Services"),
    ("AMZN", "Amazon.com Inc.", 178.30, 0.022, 0.0003, "Technology", "E-Commerce"),
    ("TSLA", "Tesla Inc.", 248.40, 0.035, 0.0001, "Consumer Cyclical", "Auto Manufacturers"),
    ("JPM", "JPMorgan Chase", 196.70, 0.014, 0.0004, "Financials", "Banking"),
    ("NVDA", "NVIDIA Corp.", 495.20, 0.030, 0.0008, "Technology", "Semiconductors"),
    ("META", "Meta Platforms", 355.60, 0.025, 0.0005, "Technology", "Social Media"),
    ("DIS", "Walt Disney Co.", 91.40, 0.019, -0.0001, "Consumer Cyclical", "Entertainment"),
    ("NFLX", "Netflix Inc.", 485.10, 0.023, 0.0006, "Consumer Cyclical", "Entertainment"),
]

# ─────────────────────────────────────────────
# Math Helpers (Black-Scholes & Greeks)
# ─────────────────────────────────────────────
def norm_cdf(x):
    """Cumulative distribution function for the standard normal distribution."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def norm_pdf(x):
    """Probability density function for the standard normal distribution."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_d1_d2(S, K, T, r, sigma):
    """Compute d1 and d2 for Black-Scholes. Returns (d1, d2)."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes option price.
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate (annual)
    sigma: Volatility (annualized)
    option_type: "call" or "put"
    """
    if T <= 0:
        if option_type == "call":
            return max(0, S - K)
        else:
            return max(0, K - S)

    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

    return max(0.01, price)  # Minimum price of 1 cent


@dataclass
class OptionGreeks:
    """Container for option Greeks."""
    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float   # per 1% move in vol


def compute_greeks(S, K, T, r, sigma, option_type="call") -> OptionGreeks:
    """
    Compute Black-Scholes Greeks.
    S: spot price, K: strike, T: time to expiry (years),
    r: risk-free rate, sigma: annualized vol, option_type: 'call'|'put'
    """
    if T <= 1e-10:
        # At expiry: delta is 1/-1 if ITM, 0 if OTM; others ~0
        if option_type == "call":
            d = 1.0 if S > K else 0.0
        else:
            d = -1.0 if S < K else 0.0
        return OptionGreeks(delta=d, gamma=0.0, theta=0.0, vega=0.0)

    d1, d2 = _bs_d1_d2(S, K, T, r, sigma)
    sqrt_T = math.sqrt(T)
    pdf_d1 = norm_pdf(d1)
    disc = math.exp(-r * T)

    # ── Gamma (same for call and put) ──
    gamma = pdf_d1 / (S * sigma * sqrt_T)

    # ── Vega (same for call and put) ──
    # Standard vega is per 1-unit change in sigma.
    # We report per 0.01 change (1 percentage-point) for readability.
    vega = S * pdf_d1 * sqrt_T * 0.01

    if option_type == "call":
        delta = norm_cdf(d1)
        # Theta per year, then convert to per calendar day (/365)
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
                 - r * K * disc * norm_cdf(d2)) / 365.0
    else:
        delta = norm_cdf(d1) - 1.0
        theta = (-(S * pdf_d1 * sigma) / (2 * sqrt_T)
                 + r * K * disc * norm_cdf(-d2)) / 365.0

    return OptionGreeks(delta=delta, gamma=gamma, theta=theta, vega=vega)


def annualized_vol(stock: "Stock") -> float:
    """Convert a stock's daily volatility to annualized."""
    return stock.volatility * math.sqrt(TRADING_DAYS_PER_YEAR)


def option_time_left(expiry_day: int, current_day: int) -> float:
    """Time to expiry in years."""
    return max(0, expiry_day - current_day) / TRADING_DAYS_PER_YEAR


def option_premium(stock: "Stock", strike: float, T: float, option_type: str) -> float:
    """Shorthand to price an option from a Stock object."""
    return black_scholes(stock.price, strike, T, RISK_FREE_RATE, annualized_vol(stock), option_type)


def option_greeks(stock: "Stock", strike: float, T: float, option_type: str) -> OptionGreeks:
    """Shorthand to compute Greeks from a Stock object."""
    return compute_greeks(stock.price, strike, T, RISK_FREE_RATE, annualized_vol(stock), option_type)


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────
@dataclass
class Stock:
    ticker: str
    name: str
    price: float
    volatility: float
    drift: float
    sector: str = "Unknown"
    industry: str = "Unknown"
    price_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.price_history:
            self.price_history = [self.price]

    def simulate_day(self):
        """Geometric Brownian Motion step for realistic price movement."""
        # GBM: dS = S * (mu*dt + sigma * sqrt(dt) * Z)
        # dt = 1 day, Z ~ N(0,1)
        z = random.gauss(0, 1)
        daily_return = self.drift + self.volatility * z
        # Clamp extreme moves to ±12% per day
        daily_return = max(-0.12, min(0.12, daily_return))
        self.price *= (1 + daily_return)
        # Floor price at $0.50
        self.price = max(0.50, round(self.price, 2))
        self.price_history.append(self.price)

    @property
    def day_change(self) -> float:
        if len(self.price_history) < 2:
            return 0.0
        return self.price_history[-1] - self.price_history[-2]

    @property
    def day_change_pct(self) -> float:
        if len(self.price_history) < 2 or self.price_history[-2] == 0:
            return 0.0
        return (self.day_change / self.price_history[-2]) * 100


@dataclass
class Holding:
    ticker: str
    shares: int
    avg_cost: float  # average cost per share

    @property
    def total_cost(self) -> float:
        return self.shares * self.avg_cost


@dataclass
class Transaction:
    day: int
    action: str  # "BUY", "SELL", "BUY_OPT", "SELL_OPT", "EXPIRE_OPT"
    ticker: str
    shares: int  # or contracts
    price: float
    total: float
    profit: float = 0.0  # Realized profit/loss for this transaction


@dataclass
class LimitOrder:
    order_id: int
    ticker: str
    order_type: str   # "BUY" or "SELL"
    shares: int
    limit_price: float
    day_placed: int

    def __str__(self):
        return (f"#{self.order_id} {self.order_type} {self.shares} {self.ticker} "
                f"@ ${self.limit_price:,.2f} (placed Day {self.day_placed})")


@dataclass
class OptionHolding:
    ticker: str
    option_type: str  # "call" or "put"
    strike: float
    expiry_day: int
    contracts: int
    avg_cost: float  # cost per contract (premium)

    @property
    def total_cost(self) -> float:
        return self.contracts * self.avg_cost * 100  # 1 contract = 100 shares


class Portfolio:
    def __init__(self, starting_balance: float):
        self.cash: float = starting_balance
        self.starting_balance: float = starting_balance
        self.holdings: Dict[str, Holding] = {}
        self.option_holdings: List[OptionHolding] = []
        self.transactions: List[Transaction] = []
        self.realized_pnl: float = 0.0
        self.pending_orders: List[LimitOrder] = []
        self._next_order_id: int = 1
        self.value_history: List[Tuple[int, float]] = []  # (day, total_value)

    def buy(self, stock: Stock, shares: int, day: int) -> Tuple[bool, str]:
        if shares <= 0:
            return False, "Number of shares must be positive."

        total_cost = shares * stock.price
        commission = 0.0  # free trades for simplicity
        total = total_cost + commission

        if total > self.cash:
            max_shares = int(self.cash / stock.price)
            return False, f"Insufficient funds. You need ${total:,.2f} but have ${self.cash:,.2f}. Max you can buy: {max_shares} shares."

        self.cash -= total

        if stock.ticker in self.holdings:
            h = self.holdings[stock.ticker]
            new_total_cost = h.total_cost + total_cost
            h.shares += shares
            h.avg_cost = new_total_cost / h.shares
        else:
            self.holdings[stock.ticker] = Holding(
                ticker=stock.ticker, shares=shares, avg_cost=stock.price
            )

        self.transactions.append(
            Transaction(day, "BUY", stock.ticker, shares, stock.price, -total, 0.0)
        )
        return True, f"Bought {shares} shares of {stock.ticker} at ${stock.price:,.2f} for ${total:,.2f}"

    def sell(self, stock: Stock, shares: int, day: int) -> Tuple[bool, str]:
        if shares <= 0:
            return False, "Number of shares must be positive."

        if stock.ticker not in self.holdings:
            return False, f"You don't own any shares of {stock.ticker}."

        h = self.holdings[stock.ticker]

        if shares > h.shares:
            return False, f"You only own {h.shares} shares of {stock.ticker}."

        total_revenue = shares * stock.price
        # Calculate cost basis for the shares being sold
        cost_basis = shares * h.avg_cost
        profit = total_revenue - cost_basis
        
        self.realized_pnl += profit
        self.cash += total_revenue

        h.shares -= shares
        if h.shares == 0:
            del self.holdings[stock.ticker]

        self.transactions.append(
            Transaction(day, "SELL", stock.ticker, shares, stock.price, total_revenue, profit)
        )

        pnl_str = f"+${profit:,.2f}" if profit >= 0 else f"-${abs(profit):,.2f}"
        return True, f"Sold {shares} shares of {stock.ticker} at ${stock.price:,.2f} for ${total_revenue:,.2f} (P&L: {pnl_str})"

    def buy_option(self, stock: Stock, option_type: str, strike: float, expiry_day: int, contracts: int, premium: float, day: int) -> Tuple[bool, str]:
        if contracts <= 0:
            return False, "Number of contracts must be positive."
            
        # 1 contract = 100 shares
        cost_per_contract = premium * 100
        total_cost = contracts * cost_per_contract
        
        if total_cost > self.cash:
            max_contracts = int(self.cash / cost_per_contract)
            return False, f"Insufficient funds. You need ${total_cost:,.2f} but have ${self.cash:,.2f}. Max you can buy: {max_contracts} contracts."
            
        self.cash -= total_cost
        
        # Add to option holdings
        self.option_holdings.append(OptionHolding(
            ticker=stock.ticker,
            option_type=option_type,
            strike=strike,
            expiry_day=expiry_day,
            contracts=contracts,
            avg_cost=premium
        ))
        
        desc = f"{stock.ticker} {expiry_day}D {strike} {option_type.upper()}"
        self.transactions.append(
            Transaction(day, f"BUY_{option_type.upper()}", desc, contracts, premium, -total_cost, 0.0)
        )
        
        return True, f"Bought {contracts} contracts of {desc} at ${premium:.2f}/share (Total: ${total_cost:,.2f})"

    def check_option_expiry(self, stocks: Dict[str, Stock], day: int) -> List[str]:
        """Check for expired options and settle them."""
        messages = []
        active_options = []
        
        for opt in self.option_holdings:
            if day >= opt.expiry_day:
                # Option is expiring today
                stock_price = stocks[opt.ticker].price
                intrinsic_value = 0.0
                
                if opt.option_type == "call":
                    intrinsic_value = max(0, stock_price - opt.strike)
                else:
                    intrinsic_value = max(0, opt.strike - stock_price)
                
                # Cash settlement
                total_value = intrinsic_value * 100 * opt.contracts
                cost_basis = opt.avg_cost * 100 * opt.contracts
                profit = total_value - cost_basis
                
                if total_value > 0:
                    self.cash += total_value
                    self.realized_pnl += profit
                    msg = f"Option EXERCISED: {opt.ticker} {opt.option_type.upper()} Strike ${opt.strike} (Exp Day {opt.expiry_day}). Payout: ${total_value:,.2f} (Profit: ${profit:,.2f})"
                    self.transactions.append(
                        Transaction(day, "EXERCISE", f"{opt.ticker} {opt.option_type}", opt.contracts, intrinsic_value, total_value, profit)
                    )
                else:
                    self.realized_pnl += profit  # Loss of premium
                    msg = f"Option EXPIRED WORTHLESS: {opt.ticker} {opt.option_type.upper()} Strike ${opt.strike} (Exp Day {opt.expiry_day}). Loss: ${abs(profit):,.2f}"
                    self.transactions.append(
                        Transaction(day, "EXPIRE", f"{opt.ticker} {opt.option_type}", opt.contracts, 0.0, 0.0, profit)
                    )
                
                messages.append(msg)
            else:
                active_options.append(opt)
        
        self.option_holdings = active_options
        return messages

    def exercise_option(self, opt_index: int, stocks: Dict[str, Stock], day: int) -> Tuple[bool, str]:
        """Early-exercise an option (American-style cash settlement)."""
        if opt_index < 0 or opt_index >= len(self.option_holdings):
            return False, "Invalid option index."

        opt = self.option_holdings[opt_index]
        stock_price = stocks[opt.ticker].price

        if opt.option_type == "call":
            intrinsic = stock_price - opt.strike
        else:
            intrinsic = opt.strike - stock_price

        if intrinsic <= 0:
            return False, f"Option is OTM (intrinsic ${intrinsic:.2f}). Cannot exercise."

        total_payout = intrinsic * 100 * opt.contracts
        cost_basis = opt.avg_cost * 100 * opt.contracts
        profit = total_payout - cost_basis

        self.cash += total_payout
        self.realized_pnl += profit
        self.option_holdings.pop(opt_index)

        desc = f"{opt.ticker} {opt.option_type.upper()} ${opt.strike}"
        self.transactions.append(
            Transaction(day, "EXERCISE", desc, opt.contracts, intrinsic, total_payout, profit)
        )
        pnl_str = f"+${profit:,.2f}" if profit >= 0 else f"-${abs(profit):,.2f}"
        return True, f"Exercised {opt.contracts}x {desc}. Intrinsic: ${intrinsic:.2f}/sh, Payout: ${total_payout:,.2f} (P&L: {pnl_str})"

    def close_option(self, opt_index: int, stocks: Dict[str, Stock], day: int) -> Tuple[bool, str]:
        """Close (sell back) an option position at current market premium."""
        if opt_index < 0 or opt_index >= len(self.option_holdings):
            return False, "Invalid option index."

        opt = self.option_holdings[opt_index]
        s = stocks[opt.ticker]
        T = option_time_left(opt.expiry_day, day)
        cur_premium = option_premium(s, opt.strike, T, opt.option_type)

        total_revenue = cur_premium * 100 * opt.contracts
        cost_basis = opt.avg_cost * 100 * opt.contracts
        profit = total_revenue - cost_basis

        self.cash += total_revenue
        self.realized_pnl += profit
        self.option_holdings.pop(opt_index)

        desc = f"{opt.ticker} {opt.option_type.upper()} ${opt.strike}"
        self.transactions.append(
            Transaction(day, "CLOSE_OPT", desc, opt.contracts, cur_premium, total_revenue, profit)
        )
        pnl_str = f"+${profit:,.2f}" if profit >= 0 else f"-${abs(profit):,.2f}"
        return True, f"Closed {opt.contracts}x {desc} at ${cur_premium:.2f}/sh. Revenue: ${total_revenue:,.2f} (P&L: {pnl_str})"

    def _option_market_value(self, opt: "OptionHolding", stocks: Dict[str, Stock], day: int) -> float:
        """Current market value of a single option holding (total, not per-share)."""
        s = stocks[opt.ticker]
        T = option_time_left(opt.expiry_day, day)
        val = option_premium(s, opt.strike, T, opt.option_type)
        return val * 100 * opt.contracts

    def get_portfolio_value(self, stocks: Dict[str, Stock], day: int = 0) -> float:
        stock_value = sum(
            h.shares * stocks[h.ticker].price for h in self.holdings.values()
        )
        opt_value = sum(
            self._option_market_value(opt, stocks, day)
            for opt in self.option_holdings
        )
        return self.cash + stock_value + opt_value

    def record_value(self, day: int, stocks: Dict[str, "Stock"]):
        """Record portfolio value for a given day, replacing any existing entry."""
        total_value = self.get_portfolio_value(stocks, day)
        if self.value_history and self.value_history[-1][0] == day:
            self.value_history[-1] = (day, total_value)
        else:
            self.value_history.append((day, total_value))

    def get_unrealized_pnl(self, stocks: Dict[str, Stock], day: int = 0) -> float:
        stock_pnl = sum(
            h.shares * (stocks[h.ticker].price - h.avg_cost)
            for h in self.holdings.values()
        )
        opt_pnl = sum(
            self._option_market_value(opt, stocks, day) - opt.total_cost
            for opt in self.option_holdings
        )
        return stock_pnl + opt_pnl

    def get_total_invested(self) -> float:
        stock_invested = sum(h.total_cost for h in self.holdings.values())
        opt_invested = sum(h.total_cost for h in self.option_holdings)
        return stock_invested + opt_invested

    # ── Limit Orders ──────────────────────────

    def place_limit_order(self, ticker: str, order_type: str, shares: int,
                          limit_price: float, day: int) -> Tuple[bool, str]:
        """Place a limit order (BUY or SELL)."""
        if shares <= 0:
            return False, "Number of shares must be positive."
        if limit_price <= 0:
            return False, "Limit price must be positive."

        if order_type == "BUY":
            # Reserve cash so the order can execute later
            total_cost = shares * limit_price
            if total_cost > self.cash:
                max_shares = int(self.cash / limit_price)
                return False, (f"Insufficient funds to reserve. Need ${total_cost:,.2f} "
                               f"but have ${self.cash:,.2f}. Max: {max_shares} shares.")
            self.cash -= total_cost  # reserve funds
        elif order_type == "SELL":
            if ticker not in self.holdings:
                return False, f"You don't own any shares of {ticker}."
            h = self.holdings[ticker]
            # Count shares already reserved by other pending sell orders
            reserved = sum(o.shares for o in self.pending_orders
                           if o.ticker == ticker and o.order_type == "SELL")
            available = h.shares - reserved
            if shares > available:
                return False, (f"Not enough available shares. You own {h.shares}, "
                               f"{reserved} reserved by other orders. Available: {available}.")
        else:
            return False, "Order type must be BUY or SELL."

        order = LimitOrder(
            order_id=self._next_order_id,
            ticker=ticker,
            order_type=order_type,
            shares=shares,
            limit_price=limit_price,
            day_placed=day,
        )
        self._next_order_id += 1
        self.pending_orders.append(order)
        return True, f"Limit order placed: {order}"

    def cancel_limit_order(self, order_id: int) -> Tuple[bool, str]:
        """Cancel a pending limit order by its ID."""
        for i, order in enumerate(self.pending_orders):
            if order.order_id == order_id:
                # Refund reserved cash for buy orders
                if order.order_type == "BUY":
                    self.cash += order.shares * order.limit_price
                self.pending_orders.pop(i)
                return True, f"Cancelled order: {order}"
        return False, f"No pending order with ID #{order_id}."

    def check_limit_orders(self, stocks: Dict[str, Stock], day: int) -> List[str]:
        """Check all pending orders against current prices. Execute those that trigger."""
        messages: List[str] = []
        remaining: List[LimitOrder] = []

        for order in self.pending_orders:
            stock = stocks.get(order.ticker)
            if stock is None:
                remaining.append(order)
                continue

            triggered = False
            if order.order_type == "BUY" and stock.price <= order.limit_price:
                triggered = True
            elif order.order_type == "SELL" and stock.price >= order.limit_price:
                triggered = True

            if triggered:
                if order.order_type == "BUY":
                    # Cash was already reserved at limit_price; execute at current market price
                    reserved = order.shares * order.limit_price
                    actual_cost = order.shares * stock.price
                    refund = reserved - actual_cost  # could be positive if price < limit
                    self.cash += refund  # give back the difference

                    if order.ticker in self.holdings:
                        h = self.holdings[order.ticker]
                        new_total = h.total_cost + actual_cost
                        h.shares += order.shares
                        h.avg_cost = new_total / h.shares
                    else:
                        self.holdings[order.ticker] = Holding(
                            ticker=order.ticker, shares=order.shares, avg_cost=stock.price
                        )

                    self.transactions.append(
                        Transaction(day, "BUY", order.ticker, order.shares, stock.price, -actual_cost, 0.0)
                    )
                    messages.append(
                        f"✅ LIMIT BUY executed: {order.shares} {order.ticker} "
                        f"@ ${stock.price:,.2f} (limit ${order.limit_price:,.2f})"
                    )
                else:  # SELL
                    if order.ticker not in self.holdings:
                        messages.append(
                            f"⚠️ LIMIT SELL cancelled: no longer own {order.ticker}"
                        )
                        continue
                    h = self.holdings[order.ticker]
                    sell_shares = min(order.shares, h.shares)
                    if sell_shares <= 0:
                        messages.append(
                            f"⚠️ LIMIT SELL cancelled: no shares of {order.ticker} left"
                        )
                        continue

                    revenue = sell_shares * stock.price
                    cost_basis = sell_shares * h.avg_cost
                    profit = revenue - cost_basis
                    self.realized_pnl += profit
                    self.cash += revenue

                    h.shares -= sell_shares
                    if h.shares == 0:
                        del self.holdings[order.ticker]

                    self.transactions.append(
                        Transaction(day, "SELL", order.ticker, sell_shares, stock.price, revenue, profit)
                    )
                    pnl_str = f"+${profit:,.2f}" if profit >= 0 else f"-${abs(profit):,.2f}"
                    messages.append(
                        f"✅ LIMIT SELL executed: {sell_shares} {order.ticker} "
                        f"@ ${stock.price:,.2f} (limit ${order.limit_price:,.2f}) P&L: {pnl_str}"
                    )
            else:
                remaining.append(order)

        self.pending_orders = remaining
        return messages


# ─────────────────────────────────────────────
# Mock News Generator
# ─────────────────────────────────────────────
class NewsGenerator:
    """Generates realistic mock financial news for stocks."""

    def reset(self):
        """Clear all cached news (used on simulation restart)."""
        self.generated_news = []
        self._news_cache = {}

    # News templates by sentiment
    POSITIVE_HEADLINES = [
        "{ticker} Beats Q{quarter} Earnings Estimates by ${beat:.2f} EPS",
        "{ticker} Announces New Partnership with {partner}",
        "{ticker} Expands into {market} Market, Stock Jumps",
        "{ticker} Receives Upgrade from {analyst} to 'Buy'",
        "{ticker} Launches Innovative {product} Product Line",
        "{ticker} Exceeds Revenue Guidance, Raises Full-Year Outlook",
        "{ticker} Secures Major Contract with {partner}",
        "{ticker} Announces Share Buyback Program of ${amount}B",
        "{ticker} Named Top Pick in {sector} Sector by {analyst}",
        "{ticker} Reports Record User Growth in {quarter}",
    ]

    NEGATIVE_HEADLINES = [
        "{ticker} Misses Q{quarter} Earnings, Shares Fall",
        "{ticker} Faces Regulatory Scrutiny Over {issue}",
        "{ticker} Downgraded by {analyst} Citing Valuation Concerns",
        "{ticker} Warns of Supply Chain Disruptions in {market}",
        "{ticker} CEO Announces Unexpected Departure",
        "{ticker} Reports Slower Growth in Key {market} Segment",
        "{ticker} Faces Increased Competition from {partner}",
        "{ticker} Cuts Full-Year Guidance Amid Market Uncertainty",
        "{ticker} Under Investigation for {issue}",
        "{ticker} Announces Layoffs Affecting {amount}% of Workforce",
    ]

    NEUTRAL_HEADLINES = [
        "{ticker} to Report Q{quarter} Earnings Next Week",
        "{ticker} Maintains Dividend at ${amount} Per Share",
        "{ticker} Announces Annual Shareholder Meeting Date",
        "{ticker} Updates {product} Platform with Minor Features",
        "{ticker} Participating in {analyst} Technology Conference",
        "{ticker} Releases Sustainability Report for {quarter}",
        "{ticker} Opens New Office in {market}",
        "{ticker} Board Approves Executive Compensation Plan",
    ]

    PARTNERS = ["Amazon", "Microsoft", "Google", "Apple", "Tesla", "Samsung", "Intel", "AMD", 
                "Oracle", "Salesforce", "IBM", "NVIDIA", "Qualcomm", "Cisco", "Adobe"]
    MARKETS = ["European", "Asian", "Emerging", "Latin American", "Indian", "Chinese", "African"]
    PRODUCTS = ["Cloud", "AI", "Mobile", "Enterprise", "Consumer", "Healthcare", "Fintech"]
    ANALYSTS = ["Goldman Sachs", "Morgan Stanley", "JP Morgan", "Bank of America", 
                "Citigroup", "Deutsche Bank", "Barclays", "UBS", "Credit Suisse"]
    ISSUES = ["Data Privacy", "Antitrust", "Environmental Compliance", "Labor Practices", 
              "Patent Infringement", "Accounting Practices"]

    def __init__(self):
        self.generated_news: List[Dict] = []
        self._news_cache: Dict[str, List[Dict]] = {}  # ticker -> news list

    def _generate_headline(self, ticker: str, sentiment: str) -> str:
        """Generate a single headline for a stock."""
        if sentiment == "positive":
            template = random.choice(self.POSITIVE_HEADLINES)
        elif sentiment == "negative":
            template = random.choice(self.NEGATIVE_HEADLINES)
        else:
            template = random.choice(self.NEUTRAL_HEADLINES)

        return template.format(
            ticker=ticker,
            quarter=random.choice([1, 2, 3, 4]),
            beat=round(random.uniform(0.05, 0.50), 2),
            partner=random.choice(self.PARTNERS),
            market=random.choice(self.MARKETS),
            analyst=random.choice(self.ANALYSTS),
            product=random.choice(self.PRODUCTS),
            amount=round(random.uniform(1, 50), 1),
            sector="Technology",
            issue=random.choice(self.ISSUES)
        )

    def generate_news(self, stocks: Dict[str, Stock], day: int, num_stories: int = 3) -> List[Dict]:
        """Generate news stories for the day."""
        news = []
        tickers = list(stocks.keys())
        
        for _ in range(num_stories):
            ticker = random.choice(tickers)
            stock = stocks[ticker]
            
            # Determine sentiment based on recent price movement
            recent_change = stock.day_change_pct if len(stock.price_history) > 1 else 0
            if recent_change > 2:
                sentiment = "positive"
            elif recent_change < -2:
                sentiment = "negative"
            else:
                sentiment = random.choice(["positive", "neutral", "negative"])
            
            headline = self._generate_headline(ticker, sentiment)
            
            story = {
                "day": day,
                "ticker": ticker,
                "headline": headline,
                "sentiment": sentiment,
                "impact": random.uniform(0.5, 3.0) if sentiment != "neutral" else 0
            }
            news.append(story)
            
            # Cache by ticker
            if ticker not in self._news_cache:
                self._news_cache[ticker] = []
            self._news_cache[ticker].append(story)
        
        self.generated_news.extend(news)
        return news

    def get_news_for_ticker(self, ticker: str, limit: int = 5) -> List[Dict]:
        """Get recent news for a specific ticker."""
        news = self._news_cache.get(ticker, [])
        return news[-limit:]

    def format_news(self, news_items: List[Dict]) -> str:
        """Format news items for display."""
        lines = []
        for item in news_items:
            sentiment_icon = {"positive": "📈", "negative": "📉", "neutral": "📰"}[item["sentiment"]]
            lines.append(f"  {sentiment_icon} Day {item['day']}: {item['headline']}")
        return "\n".join(lines) if lines else "  No recent news."


# ─────────────────────────────────────────────
# Portfolio Analyzer
# ─────────────────────────────────────────────
class PortfolioAnalyzer:
    """Analyzes portfolio composition, risk, and provides recommendations."""

    # Risk thresholds
    CONCENTRATION_WARNING = 0.30  # 30% in single position
    CONCENTRATION_CRITICAL = 0.50  # 50% in single position
    SECTOR_WARNING = 0.50  # 50% in single sector
    SECTOR_CRITICAL = 0.70  # 70% in single sector
    VOLATILITY_HIGH = 0.025  # 2.5% daily volatility
    DRAWDOWN_WARNING = -0.15  # 15% drawdown
    DRAWDOWN_CRITICAL = -0.25  # 25% drawdown
    SHARED_NEWS = NewsGenerator()

    # Sector risk profiles
    SECTOR_RISKS = {
        "Technology": "high",
        "Consumer Cyclical": "medium-high",
        "Financials": "medium",
        "Healthcare": "low-medium",
        "Energy": "high",
        "Utilities": "low",
        "Consumer Defensive": "low",
        "Industrials": "medium",
        "Materials": "medium-high",
        "Communication Services": "medium",
        "Real Estate": "medium",
    }

    def __init__(self, portfolio: Portfolio, stocks: Dict[str, Stock]):
        self.portfolio = portfolio
        self.stocks = stocks
        self.news_gen = self.SHARED_NEWS

    def analyze(self, day: int) -> Dict:
        """Run complete portfolio analysis."""
        return {
            "overview": self._analyze_overview(day),
            "distribution": self._analyze_distribution(day),
            "sector_allocation": self._analyze_sector_allocation(day),
            "risk_metrics": self._calculate_risk_metrics(day),
            "position_analysis": self._analyze_positions(day),
            "warnings": self._generate_warnings(day),
            "recommendations": self._generate_recommendations(day),
            "news": self.news_gen.generate_news(self.stocks, day, 3)
        }

    def _analyze_overview(self, day: int) -> Dict:
        """Generate portfolio overview statistics."""
        total_value = self.portfolio.get_portfolio_value(self.stocks, day)
        cash_pct = self.portfolio.cash / total_value if total_value > 0 else 0
        
        stock_value = sum(
            h.shares * self.stocks[h.ticker].price 
            for h in self.portfolio.holdings.values()
        )
        
        opt_value = sum(
            self.portfolio._option_market_value(opt, self.stocks, day)
            for opt in self.portfolio.option_holdings
        )
        
        return {
            "total_value": total_value,
            "cash": self.portfolio.cash,
            "cash_pct": cash_pct,
            "stock_value": stock_value,
            "stock_pct": stock_value / total_value if total_value > 0 else 0,
            "option_value": opt_value,
            "option_pct": opt_value / total_value if total_value > 0 else 0,
            "num_positions": len(self.portfolio.holdings),
            "num_options": len(self.portfolio.option_holdings),
            "realized_pnl": self.portfolio.realized_pnl,
            "unrealized_pnl": self.portfolio.get_unrealized_pnl(self.stocks, day),
            "total_pnl": total_value - self.portfolio.starting_balance,
            "roi_pct": ((total_value / self.portfolio.starting_balance) - 1) * 100
        }

    def _analyze_distribution(self, day: int) -> Dict:
        """Analyze position size distribution."""
        total_value = self.portfolio.get_portfolio_value(self.stocks, day)
        if total_value == 0:
            return {"positions": [], "largest_position": None}
        
        positions = []
        for ticker, holding in self.portfolio.holdings.items():
            stock = self.stocks[ticker]
            value = holding.shares * stock.price
            pct = value / total_value
            pnl = value - holding.total_cost
            pnl_pct = (pnl / holding.total_cost * 100) if holding.total_cost > 0 else 0
            
            positions.append({
                "ticker": ticker,
                "name": stock.name,
                "sector": stock.sector,
                "industry": stock.industry,
                "shares": holding.shares,
                "value": value,
                "pct": pct,
                "avg_cost": holding.avg_cost,
                "current_price": stock.price,
                "pnl": pnl,
                "pnl_pct": pnl_pct
            })
        
        # Add options as positions
        for opt in self.portfolio.option_holdings:
            stock = self.stocks[opt.ticker]
            value = self.portfolio._option_market_value(opt, self.stocks, day)
            pct = value / total_value if total_value > 0 else 0
            pnl = value - opt.total_cost
            
            positions.append({
                "ticker": f"{opt.ticker} {opt.option_type.upper()}",
                "name": f"{opt.strike} Strike (Exp {opt.expiry_day})",
                "sector": f"Option ({stock.sector})",
                "industry": "Derivative",
                "contracts": opt.contracts,
                "value": value,
                "pct": pct,
                "avg_cost": opt.avg_cost * 100,
                "current_price": self._get_opt_premium(opt, day),
                "pnl": pnl,
                "pnl_pct": (pnl / opt.total_cost * 100) if opt.total_cost > 0 else 0
            })
        
        positions.sort(key=lambda x: x["value"], reverse=True)
        
        return {
            "positions": positions,
            "largest_position": positions[0] if positions else None,
            "num_positions": len(positions)
        }

    def _get_opt_premium(self, opt: OptionHolding, day: int) -> float:
        """Get current premium for an option."""
        stock = self.stocks[opt.ticker]
        T = option_time_left(opt.expiry_day, day)
        return option_premium(stock, opt.strike, T, opt.option_type)

    def _analyze_sector_allocation(self, day: int) -> Dict:
        """Analyze allocation by sector."""
        total_value = self.portfolio.get_portfolio_value(self.stocks, day)
        if total_value == 0:
            return {"sectors": {}, "largest_sector": None}
        
        sectors = {}
        
        # Stock allocations by sector
        for ticker, holding in self.portfolio.holdings.items():
            stock = self.stocks[ticker]
            value = holding.shares * stock.price
            if stock.sector not in sectors:
                sectors[stock.sector] = {"value": 0, "stocks": [], "risk": self.SECTOR_RISKS.get(stock.sector, "medium")}
            sectors[stock.sector]["value"] += value
            sectors[stock.sector]["stocks"].append(ticker)
        
        # Option allocations (attribute to underlying sector)
        for opt in self.portfolio.option_holdings:
            stock = self.stocks[opt.ticker]
            value = self.portfolio._option_market_value(opt, self.stocks, day)
            sector_key = f"{stock.sector} (Options)"
            if sector_key not in sectors:
                sectors[sector_key] = {"value": 0, "stocks": [], "risk": self.SECTOR_RISKS.get(stock.sector, "medium")}
            sectors[sector_key]["value"] += value
            sectors[sector_key]["stocks"].append(f"{opt.ticker} {opt.option_type}")
        
        # Calculate percentages
        for sector in sectors:
            sectors[sector]["pct"] = sectors[sector]["value"] / total_value
        
        # Find largest sector
        largest = max(sectors.items(), key=lambda x: x[1]["value"]) if sectors else None
        
        return {
            "sectors": sectors,
            "largest_sector": largest,
            "num_sectors": len(set(s.replace(" (Options)", "") for s in sectors.keys()))
        }

    def _calculate_risk_metrics(self, day: int) -> Dict:
        """Calculate various risk metrics."""
        total_value = self.portfolio.get_portfolio_value(self.stocks, day)
        
        # Portfolio volatility (weighted average of position volatilities)
        weights = []
        vols = []
        for ticker, holding in self.portfolio.holdings.items():
            stock = self.stocks[ticker]
            weight = (holding.shares * stock.price) / total_value if total_value > 0 else 0
            weights.append(weight)
            vols.append(stock.volatility)
        
        portfolio_vol = sum(w * v for w, v in zip(weights, vols)) if weights else 0
        
        # Beta approximation (simplified)
        portfolio_beta = 1.0  # Assume market beta for simplicity
        
        # Max drawdown from peak using recorded history
        if self.portfolio.value_history:
            peak_value = self.portfolio.value_history[0][1]
            max_drawdown = 0.0
            for _, value in self.portfolio.value_history:
                if value > peak_value:
                    peak_value = value
                drawdown = (value - peak_value) / peak_value if peak_value > 0 else 0
                max_drawdown = min(max_drawdown, drawdown)
        else:
            max_drawdown = 0.0
        
        current_pnl_pct = (total_value - self.portfolio.starting_balance) / self.portfolio.starting_balance
        
        return {
            "portfolio_volatility": portfolio_vol,
            "annualized_volatility": portfolio_vol * math.sqrt(TRADING_DAYS_PER_YEAR),
            "portfolio_beta": portfolio_beta,
            "current_drawdown": current_pnl_pct if current_pnl_pct < 0 else 0,
            "max_drawdown": max_drawdown,
            "value_at_risk_95": total_value * portfolio_vol * 1.65  # 95% VaR (simplified)
        }

    def _analyze_positions(self, day: int) -> List[Dict]:
        """Analyze individual positions with pros/cons."""
        analyses = []
        
        for ticker, holding in self.portfolio.holdings.items():
            stock = self.stocks[ticker]
            analysis = self._analyze_single_position(ticker, holding, stock, day)
            analyses.append(analysis)
        
        for opt in self.portfolio.option_holdings:
            stock = self.stocks[opt.ticker]
            analysis = self._analyze_option_position(opt, stock, day)
            analyses.append(analysis)
        
        return analyses

    def _analyze_single_position(self, ticker: str, holding: Holding, stock: Stock, day: int) -> Dict:
        """Analyze a single stock position."""
        current_price = stock.price
        avg_cost = holding.avg_cost
        pnl_pct = ((current_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0
        
        # Calculate price momentum
        if len(stock.price_history) >= 5:
            momentum_5d = (current_price - stock.price_history[-5]) / stock.price_history[-5] * 100
        else:
            momentum_5d = 0
        
        if len(stock.price_history) >= 10:
            momentum_10d = (current_price - stock.price_history[-10]) / stock.price_history[-10] * 100
        else:
            momentum_10d = 0
        
        pros = []
        cons = []
        
        # Profitability analysis
        if pnl_pct > 10:
            pros.append(f"Strong unrealized gains (+{pnl_pct:.1f}%)")
        elif pnl_pct > 0:
            pros.append(f"Currently profitable (+{pnl_pct:.1f}%)")
        elif pnl_pct < -10:
            cons.append(f"Significant unrealized loss ({pnl_pct:.1f}%)")
        elif pnl_pct < 0:
            cons.append(f"Small unrealized loss ({pnl_pct:.1f}%)")
        
        # Momentum analysis
        if momentum_5d > 3:
            pros.append(f"Positive 5-day momentum (+{momentum_5d:.1f}%)")
        elif momentum_5d < -3:
            cons.append(f"Negative 5-day momentum ({momentum_5d:.1f}%)")
        
        if momentum_10d > 5:
            pros.append(f"Strong 10-day uptrend (+{momentum_10d:.1f}%)")
        elif momentum_10d < -5:
            cons.append(f"Declining 10-day trend ({momentum_10d:.1f}%)")
        
        # Volatility analysis
        if stock.volatility < 0.015:
            pros.append("Low volatility - stable investment")
        elif stock.volatility > 0.03:
            cons.append("High volatility - higher risk")
        
        # Sector analysis
        sector_risk = self.SECTOR_RISKS.get(stock.sector, "medium")
        if sector_risk == "low":
            pros.append(f"Defensive {stock.sector} sector")
        elif sector_risk == "high":
            cons.append(f"Cyclical {stock.sector} sector - higher risk")
        
        # Price relative to history
        if len(stock.price_history) > 1:
            hist_high = max(stock.price_history)
            hist_low = min(stock.price_history)
            if current_price >= hist_high * 0.95:
                cons.append("Trading near historical high - consider taking profits")
            elif current_price <= hist_low * 1.05:
                pros.append("Trading near historical low - potential value opportunity")
        
        return {
            "type": "stock",
            "ticker": ticker,
            "name": stock.name,
            "sector": stock.sector,
            "shares": holding.shares,
            "avg_cost": avg_cost,
            "current_price": current_price,
            "pnl": holding.shares * (current_price - avg_cost),
            "pnl_pct": pnl_pct,
            "momentum_5d": momentum_5d,
            "momentum_10d": momentum_10d,
            "volatility": stock.volatility,
            "pros": pros,
            "cons": cons,
            "recommendation": self._position_recommendation(pros, cons, pnl_pct)
        }

    def _analyze_option_position(self, opt: OptionHolding, stock: Stock, day: int) -> Dict:
        """Analyze an option position."""
        T = option_time_left(opt.expiry_day, day)
        dte = max(0, opt.expiry_day - day)
        current_premium = option_premium(stock, opt.strike, T, opt.option_type)
        greeks = option_greeks(stock, opt.strike, T, opt.option_type)
        
        mkt_value = current_premium * 100 * opt.contracts
        cost_basis = opt.avg_cost * 100 * opt.contracts
        pnl = mkt_value - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
        
        # Moneyness
        if opt.option_type == "call":
            intrinsic = max(0, stock.price - opt.strike)
            moneyness = "ITM" if stock.price > opt.strike else ("ATM" if abs(stock.price - opt.strike) < 5 else "OTM")
        else:
            intrinsic = max(0, opt.strike - stock.price)
            moneyness = "ITM" if stock.price < opt.strike else ("ATM" if abs(stock.price - opt.strike) < 5 else "OTM")
        
        pros = []
        cons = []
        
        # Time decay warning
        if dte <= 5:
            cons.append(f"⚠️ CRITICAL: Only {dte} days to expiry - rapid time decay")
        elif dte <= 10:
            cons.append(f"⚠️ WARNING: Only {dte} days to expiry - accelerating time decay")
        elif dte <= 20:
            cons.append(f"Time decay accelerating ({dte} DTE)")
        else:
            pros.append(f"Adequate time remaining ({dte} DTE)")
        
        # Moneyness
        if moneyness == "ITM":
            pros.append(f"In-the-money with ${intrinsic:.2f} intrinsic value")
        elif moneyness == "ATM":
            pros.append("At-the-money - high delta sensitivity")
            cons.append("At-the-money - high gamma risk")
        else:
            cons.append("Out-of-the-money - needs price movement")
        
        # Profitability
        if pnl_pct > 50:
            pros.append(f"Excellent returns (+{pnl_pct:.1f}%) - consider taking profits")
        elif pnl_pct > 0:
            pros.append(f"Currently profitable (+{pnl_pct:.1f}%)")
        elif pnl_pct < -50:
            cons.append(f"Significant loss ({pnl_pct:.1f}%) - consider cutting losses")
        
        # Greeks analysis
        if abs(greeks.delta) > 0.7:
            pros.append("High delta - behaves like stock")
        if greeks.theta < -0.5:
            cons.append(f"High theta decay ({greeks.theta:.3f}/day)")
        
        return {
            "type": "option",
            "ticker": opt.ticker,
            "option_type": opt.option_type,
            "strike": opt.strike,
            "expiry_day": opt.expiry_day,
            "dte": dte,
            "contracts": opt.contracts,
            "moneyness": moneyness,
            "intrinsic": intrinsic,
            "current_premium": current_premium,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "delta": greeks.delta,
            "gamma": greeks.gamma,
            "theta": greeks.theta,
            "vega": greeks.vega,
            "pros": pros,
            "cons": cons,
            "recommendation": self._option_recommendation(opt, dte, moneyness, pnl_pct, greeks)
        }

    def _position_recommendation(self, pros: List[str], cons: List[str], pnl_pct: float) -> str:
        """Generate recommendation for a stock position."""
        if pnl_pct > 20 and len(cons) > len(pros):
            return "HOLD/REDUCE - Take some profits given concerns"
        elif pnl_pct < -15:
            return "REVIEW - Consider cutting losses"
        elif len(pros) > len(cons) + 1:
            return "HOLD/BUY - Favorable outlook"
        elif len(cons) > len(pros) + 1:
            return "REDUCE - Multiple concerns"
        else:
            return "HOLD - Monitor closely"

    def _option_recommendation(self, opt: OptionHolding, dte: int, moneyness: str, pnl_pct: float, greeks: OptionGreeks) -> str:
        """Generate recommendation for an option position."""
        if dte <= 5:
            if moneyness == "ITM":
                return "EXERCISE or CLOSE - Expiring soon, capture intrinsic value"
            elif pnl_pct < -50:
                return "LET EXPIRE - Minimal recovery expected"
            else:
                return "CLOSE - Time decay is extreme"
        
        if pnl_pct > 100:
            return "CLOSE/ROLL - Excellent gains, consider taking profits"
        
        if moneyness == "OTM" and dte < 20:
            return "CLOSE - OTM with limited time, low probability"
        
        if pnl_pct < -50 and dte < 30:
            return "CLOSE - Cut losses, unlikely to recover"
        
        return "HOLD - Monitor for changes"

    def _generate_warnings(self, day: int) -> List[Dict]:
        """Generate risk warnings for the portfolio."""
        warnings = []
        total_value = self.portfolio.get_portfolio_value(self.stocks, day)
        
        if total_value == 0:
            return warnings
        
        # Check position concentration
        dist = self._analyze_distribution(day)
        if dist["largest_position"]:
            largest = dist["largest_position"]
            if largest["pct"] > self.CONCENTRATION_CRITICAL:
                warnings.append({
                    "level": "CRITICAL",
                    "type": "concentration",
                    "message": f"CRITICAL: {largest['ticker']} represents {largest['pct']*100:.1f}% of portfolio. Severe concentration risk!",
                    "explanation": "Having over 50% in a single position exposes you to significant idiosyncratic risk. If this stock declines sharply, your entire portfolio suffers disproportionately. Consider diversifying into other sectors."
                })
            elif largest["pct"] > self.CONCENTRATION_WARNING:
                warnings.append({
                    "level": "WARNING",
                    "type": "concentration",
                    "message": f"WARNING: {largest['ticker']} is {largest['pct']*100:.1f}% of portfolio. High concentration.",
                    "explanation": "Positions over 30% create concentration risk. Consider trimming this position and reinvesting in underrepresented sectors to improve diversification."
                })
        
        # Check sector concentration
        sectors = self._analyze_sector_allocation(day)
        if sectors["largest_sector"]:
            sector_name, sector_data = sectors["largest_sector"]
            base_sector = sector_name.replace(" (Options)", "")
            if sector_data["pct"] > self.SECTOR_CRITICAL:
                warnings.append({
                    "level": "CRITICAL",
                    "type": "sector",
                    "message": f"CRITICAL: {base_sector} sector is {sector_data['pct']*100:.1f}% of portfolio!",
                    "explanation": f"Extreme sector concentration makes your portfolio vulnerable to sector-specific downturns. {base_sector} stocks often move together - a sector rotation or regulatory change could significantly impact your entire portfolio."
                })
            elif sector_data["pct"] > self.SECTOR_WARNING:
                warnings.append({
                    "level": "WARNING",
                    "type": "sector",
                    "message": f"WARNING: {base_sector} sector concentration at {sector_data['pct']*100:.1f}%",
                    "explanation": f"Over 50% in {base_sector} reduces diversification benefits. Consider adding exposure to defensive sectors like Consumer Defensive or Utilities to balance your portfolio."
                })
        
        # Check cash levels
        cash_pct = self.portfolio.cash / total_value
        if cash_pct > 0.50:
            warnings.append({
                "level": "INFO",
                "type": "cash",
                "message": f"High cash position: {cash_pct*100:.1f}% uninvested",
                "explanation": "While cash provides safety, holding over 50% in cash may lead to opportunity cost. Consider dollar-cost averaging into quality positions or adding to existing winners."
            })
        elif cash_pct < 0.05:
            warnings.append({
                "level": "WARNING",
                "type": "cash",
                "message": f"Low cash: Only {cash_pct*100:.1f}% available",
                "explanation": "With less than 5% cash, you have limited dry powder for opportunities or emergencies. Consider building a cash reserve of at least 10-20%."
            })
        
        # Check for options expiring soon
        for opt in self.portfolio.option_holdings:
            dte = max(0, opt.expiry_day - day)
            if dte <= 5:
                warnings.append({
                    "level": "CRITICAL",
                    "type": "options",
                    "message": f"URGENT: {opt.ticker} {opt.option_type.upper()} ${opt.strike} expires in {dte} days!",
                    "explanation": "Options lose value rapidly near expiry (theta decay). If ITM, consider exercising. If OTM, consider closing to salvage remaining value."
                })
        
        # Check portfolio performance
        total_pnl = total_value - self.portfolio.starting_balance
        pnl_pct = total_pnl / self.portfolio.starting_balance
        if pnl_pct < self.DRAWDOWN_CRITICAL:
            warnings.append({
                "level": "CRITICAL",
                "type": "performance",
                "message": f"Portfolio down {pnl_pct*100:.1f}% - Significant drawdown",
                "explanation": "Your portfolio has experienced a severe drawdown. Review your risk management strategy. Consider reducing position sizes and avoiding high-beta stocks until market conditions improve."
            })
        elif pnl_pct < self.DRAWDOWN_WARNING:
            warnings.append({
                "level": "WARNING",
                "type": "performance",
                "message": f"Portfolio down {pnl_pct*100:.1f}% - Monitor closely",
                "explanation": "Your portfolio is in a moderate drawdown. Review underperforming positions and ensure your stop-losses are in place."
            })
        
        return warnings

    def _generate_recommendations(self, day: int) -> List[Dict]:
        """Generate portfolio-level recommendations."""
        recommendations = []
        total_value = self.portfolio.get_portfolio_value(self.stocks, day)
        
        if total_value == 0:
            return [{"title": "Start Investing", "action": "Begin building positions in diversified sectors"}]
        
        dist = self._analyze_distribution(day)
        sectors = self._analyze_sector_allocation(day)
        
        # Diversification recommendations
        if sectors["num_sectors"] < 3:
            recommendations.append({
                "title": "Improve Sector Diversification",
                "priority": "HIGH",
                "action": f"Your portfolio spans only {sectors['num_sectors']} sectors. Consider adding exposure to underrepresented sectors to reduce correlation risk."
            })
        
        # Position sizing recommendations
        if dist["largest_position"] and dist["largest_position"]["pct"] > 0.25:
            recommendations.append({
                "title": "Reduce Concentration Risk",
                "priority": "HIGH",
                "action": f"Trim {dist['largest_position']['ticker']} position from {dist['largest_position']['pct']*100:.1f}% to under 20%. Reinvest proceeds across 2-3 different sectors."
            })
        
        # Cash management
        cash_pct = self.portfolio.cash / total_value
        if cash_pct > 0.30:
            recommendations.append({
                "title": "Deploy Excess Cash",
                "priority": "MEDIUM",
                "action": f"You have {cash_pct*100:.1f}% in cash. Consider deploying 10-15% into quality dividend stocks or adding to existing winners on dips."
            })
        
        # Profit taking
        big_winners = [p for p in dist["positions"] if p.get("pnl_pct", 0) > 30]
        if len(big_winners) >= 2:
            recommendations.append({
                "title": "Consider Taking Profits",
                "priority": "MEDIUM",
                "action": f"You have {len(big_winners)} positions with >30% gains. Consider trimming 20-30% of these positions to lock in profits and reduce risk."
            })
        
        # Loss management
        big_losers = [p for p in dist["positions"] if p.get("pnl_pct", 0) < -20]
        if len(big_losers) >= 1:
            recommendations.append({
                "title": "Review Losing Positions",
                "priority": "HIGH",
                "action": f"{len(big_losers)} position(s) down >20%. Re-evaluate thesis. If fundamentals changed, consider cutting losses. If not, consider averaging down carefully."
            })
        
        # Options recommendations
        if not self.portfolio.option_holdings:
            recommendations.append({
                "title": "Consider Options for Income/Hedging",
                "priority": "LOW",
                "action": "You have no options positions. Consider covered calls on large stock positions for income, or protective puts for downside protection."
            })
        
        # Rebalancing suggestion
        if len(self.portfolio.holdings) >= 3:
            recommendations.append({
                "title": "Periodic Rebalancing",
                "priority": "MEDIUM",
                "action": "Review portfolio quarterly. Trim winners that exceed target allocation, add to quality positions that have underperformed."
            })
        
        return recommendations


# ─────────────────────────────────────────────
# CLI Chart Renderer
# ─────────────────────────────────────────────
def render_chart(stock: Stock, num_days: int = CHART_WIDTH, height: int = CHART_HEIGHT) -> str:
    """Render an ASCII chart of the last `num_days` of price history."""
    history = stock.price_history[-num_days:]
    if len(history) < 2:
        return "  Not enough data to render chart.\n"

    min_price = min(history)
    max_price = max(history)
    price_range = max_price - min_price

    if price_range == 0:
        price_range = max_price * 0.01  # avoid division by zero

    lines = []
    header = f"  {stock.ticker} — Last {len(history)} days  |  Range: ${min_price:,.2f} – ${max_price:,.2f}"
    lines.append(header)
    lines.append("  " + "─" * (len(history) * 6 + 10))

    # Build the grid
    grid = [[" " for _ in range(len(history))] for _ in range(height)]

    for col, price in enumerate(history):
        row = int((price - min_price) / price_range * (height - 1))
        row = min(row, height - 1)
        grid[row][col] = "●"

    # Connect points with lines
    for col in range(len(history) - 1):
        row1 = int((history[col] - min_price) / price_range * (height - 1))
        row2 = int((history[col + 1] - min_price) / price_range * (height - 1))
        row1 = min(row1, height - 1)
        row2 = min(row2, height - 1)
        if abs(row2 - row1) > 1:
            step = 1 if row2 > row1 else -1
            for r in range(row1 + step, row2, step):
                if grid[r][col] == " ":
                    grid[r][col] = "│"

    # Render top to bottom
    for r in range(height - 1, -1, -1):
        price_at_row = min_price + (r / (height - 1)) * price_range
        row_label = f"${price_at_row:>8.2f} │"
        row_content = "  ".join(f" {grid[r][c]} " for c in range(len(history)))
        lines.append(f"  {row_label} {row_content}")

    # X-axis
    axis_padding = " " * 11
    x_labels = "  ".join(f"D{len(stock.price_history) - len(history) + i:>2}" for i in range(len(history)))
    lines.append(f"  {axis_padding}{'─' * (len(history) * 5 + 2)}")
    lines.append(f"  {axis_padding} {x_labels}")

    # Price annotations
    start_p = history[0]
    end_p = history[-1]
    change = end_p - start_p
    change_pct = (change / start_p * 100) if start_p != 0 else 0
    arrow = "▲" if change >= 0 else "▼"
    color_code = "\033[92m" if change >= 0 else "\033[91m"
    reset = "\033[0m"
    lines.append(f"  {color_code}{arrow} {abs(change):+.2f} ({change_pct:+.2f}%) over shown period{reset}")
    lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Display Helpers
# ─────────────────────────────────────────────
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def colored(text: str, color: str) -> str:
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "cyan": "\033[96m",
        "magenta": "\033[95m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "reset": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors.get('reset', '')}"


def format_change(value: float, is_pct: bool = False) -> str:
    suffix = "%" if is_pct else ""
    if value > 0:
        return colored(f"+{value:,.2f}{suffix}", "green")
    elif value < 0:
        return colored(f"{value:,.2f}{suffix}", "red")
    else:
        return f"{value:,.2f}{suffix}"


def format_money(value: float) -> str:
    if value >= 0:
        return colored(f"${value:,.2f}", "green")
    else:
        return colored(f"-${abs(value):,.2f}", "red")


def print_header(day: int, portfolio: Portfolio, stocks: Dict[str, Stock]):
    total_value = portfolio.get_portfolio_value(stocks, day)
    total_pnl = total_value - portfolio.starting_balance
    pnl_pct = (total_pnl / portfolio.starting_balance) * 100

    print(colored("╔══════════════════════════════════════════════════════════════════════╗", "cyan"))
    print(colored("║", "cyan") + colored("          📈  STOCK MARKET SIMULATOR  📉", "bold").center(79) + colored("║", "cyan"))
    print(colored("╠══════════════════════════════════════════════════════════════════════╣", "cyan"))
    print(colored("║", "cyan") + f"  Day {day}/{MAX_DAYS}  │  Cash: ${portfolio.cash:>10,.2f}  │  Portfolio: ${total_value:>10,.2f}  │  P&L: {format_change(total_pnl)} ({format_change(pnl_pct, True)})  " + colored("║", "cyan"))
    print(colored("╚══════════════════════════════════════════════════════════════════════╝", "cyan"))
    print()


def print_market_table(stocks: Dict[str, Stock], day: int):
    print(colored("  ┌─────┬────────────────────┬───────────┬───────────┬──────────┐", "dim"))
    print(colored("  │", "dim") + colored(" #   ", "bold") +
          colored("│", "dim") + colored(" Ticker / Name      ", "bold") +
          colored("│", "dim") + colored("   Price   ", "bold") +
          colored("│", "dim") + colored("  Change   ", "bold") +
          colored("│", "dim") + colored(" Change % ", "bold") +
          colored("│", "dim"))
    print(colored("  ├─────┼────────────────────┼───────────┼───────────┼──────────┤", "dim"))

    for i, (ticker, stock) in enumerate(stocks.items(), 1):
        change = stock.day_change
        change_pct = stock.day_change_pct
        change_str = format_change(change)
        change_pct_str = format_change(change_pct, True)

        print(colored("  │", "dim") +
              f" {i:<3} " +
              colored("│", "dim") +
              f" {ticker:<18} " +
              colored("│", "dim") +
              f" ${stock.price:>8.2f}" +
              colored("│", "dim") +
              f" {change_str:>18}" +
              colored("│", "dim") +
              f" {change_pct_str:>17}" +
              colored("│", "dim"))

    print(colored("  └─────┴────────────────────┴───────────┴───────────┴──────────┘", "dim"))
    print()


def print_portfolio_detail(portfolio: Portfolio, stocks: Dict[str, Stock], day: int):
    print(colored("\n  ═══════════════════════════════════════════════════════", "yellow"))
    print(colored("                    YOUR PORTFOLIO", "bold"))
    print(colored("  ═══════════════════════════════════════════════════════", "yellow"))

    if not portfolio.holdings and not portfolio.option_holdings:
        print("  You don't own any stocks or options yet.\n")
    
    if portfolio.holdings:
        print(colored("  STOCKS:", "bold"))
        print(colored("  ┌────────┬────────┬───────────┬───────────┬───────────┬───────────┐", "dim"))
        print(colored("  │", "dim") + colored(" Ticker ", "bold") +
              colored("│", "dim") + colored(" Shares ", "bold") +
              colored("│", "dim") + colored(" Avg Cost  ", "bold") +
              colored("│", "dim") + colored(" Cur Price ", "bold") +
              colored("│", "dim") + colored(" Mkt Value ", "bold") +
              colored("│", "dim") + colored("   P&L     ", "bold") +
              colored("│", "dim"))
        print(colored("  ├────────┼────────┼───────────┼───────────┼───────────┼───────────┤", "dim"))

        for ticker, h in sorted(portfolio.holdings.items()):
            cur_price = stocks[ticker].price
            mkt_value = h.shares * cur_price
            pnl = h.shares * (cur_price - h.avg_cost)
            pnl_display = format_change(pnl)

            print(colored("  │", "dim") +
                  f" {ticker:<6} " +
                  colored("│", "dim") +
                  f" {h.shares:>6} " +
                  colored("│", "dim") +
                  f" ${h.avg_cost:>8.2f}" +
                  colored("│", "dim") +
                  f" ${cur_price:>8.2f}" +
                  colored("│", "dim") +
                  f" ${mkt_value:>8.2f}" +
                  colored("│", "dim") +
                  f" {pnl_display:>18}" +
                  colored("│", "dim"))

        print(colored("  └────────┴────────┴───────────┴───────────┴───────────┴───────────┘", "dim"))
        print()

    if portfolio.option_holdings:
        print(colored("  OPTIONS:", "bold"))
        print(colored("  ┌───────────────────┬─────┬──────┬──────────┬──────────┬──────────┬────────┬────────┬────────┬────────┐", "dim"))
        print(colored("  │", "dim") + colored(" Contract          ", "bold") +
              colored("│", "dim") + colored(" Qty ", "bold") +
              colored("│", "dim") + colored(" DTE  ", "bold") +
              colored("│", "dim") + colored(" Avg Prem ", "bold") +
              colored("│", "dim") + colored(" Cur Prem ", "bold") +
              colored("│", "dim") + colored("   P&L    ", "bold") +
              colored("│", "dim") + colored(" Delta  ", "bold") +
              colored("│", "dim") + colored(" Gamma  ", "bold") +
              colored("│", "dim") + colored(" Theta  ", "bold") +
              colored("│", "dim") + colored(" Vega   ", "bold") +
              colored("│", "dim"))
        print(colored("  ├───────────────────┼─────┼──────┼──────────┼──────────┼──────────┼────────┼────────┼────────┼────────┤", "dim"))

        for opt in portfolio.option_holdings:
            s = stocks[opt.ticker]
            T = option_time_left(opt.expiry_day, day)
            cur_prem = option_premium(s, opt.strike, T, opt.option_type)
            g = option_greeks(s, opt.strike, T, opt.option_type)

            mkt_value = cur_prem * 100 * opt.contracts
            cost_basis = opt.avg_cost * 100 * opt.contracts
            pnl = mkt_value - cost_basis
            dte = max(0, opt.expiry_day - day)

            desc = f"{opt.ticker} ${opt.strike:.0f} {opt.option_type.upper()[:1]}"

            print(colored("  │", "dim") +
                  f" {desc:<17} " +
                  colored("│", "dim") +
                  f" {opt.contracts:>3} " +
                  colored("│", "dim") +
                  f" {dte:>4} " +
                  colored("│", "dim") +
                  f" ${opt.avg_cost:>7.2f} " +
                  colored("│", "dim") +
                  f" ${cur_prem:>7.2f} " +
                  colored("│", "dim") +
                  f" {format_change(pnl):>17}" +
                  colored("│", "dim") +
                  f" {g.delta:>+5.3f} " +
                  colored("│", "dim") +
                  f" {g.gamma:>6.4f}" +
                  colored("│", "dim") +
                  f" {g.theta:>6.3f}" +
                  colored("│", "dim") +
                  f" {g.vega:>6.3f}" +
                  colored("│", "dim"))

        print(colored("  └───────────────────┴─────┴──────┴──────────┴──────────┴──────────┴────────┴────────┴────────┴────────┘", "dim"))


    # Summary
    total_value = portfolio.get_portfolio_value(stocks, day)
    unrealized = portfolio.get_unrealized_pnl(stocks, day)
    total_pnl = total_value - portfolio.starting_balance
    total_pnl_pct = (total_pnl / portfolio.starting_balance) * 100
    invested = portfolio.get_total_invested()

    print()
    print(f"  {'Cash Balance:':<25} ${portfolio.cash:>12,.2f}")
    print(f"  {'Invested Value:':<25} ${invested:>12,.2f}")
    print(f"  {'Total Portfolio Value:':<25} ${total_value:>12,.2f}")
    print(f"  {'─' * 40}")
    print(f"  {'Unrealized P&L:':<25} {format_money(unrealized):>23}")
    print(f"  {'Realized P&L:':<25} {format_money(portfolio.realized_pnl):>23}")
    print(f"  {'Total P&L:':<25} {format_money(total_pnl):>23} ({format_change(total_pnl_pct, True)})")
    print(f"  {'─' * 40}")

    # Win/Loss ratio from transactions
    # Wins are realized trades with profit > 0
    _close_actions = {"SELL", "EXERCISE", "CLOSE_OPT", "EXPIRE"}
    wins = sum(1 for t in portfolio.transactions if t.action in _close_actions and t.profit > 0)
    losses = sum(1 for t in portfolio.transactions if t.action in _close_actions and t.profit <= 0)
    
    print(f"  {'Winning Trades:':<25} {wins:>12}")
    print(f"  {'Losing Trades:':<25} {losses:>12}")

    # Return on investment
    roi = (total_value / portfolio.starting_balance - 1) * 100
    print(f"  {'Return on Investment:':<25} {format_change(roi, True):>23}")
    print()


def print_transaction_history(portfolio: Portfolio):
    print(colored("\n  ═══════════════════════════════════════════════════════", "yellow"))
    print(colored("                  TRANSACTION HISTORY", "bold"))
    print(colored("  ═══════════════════════════════════════════════════════", "yellow"))

    if not portfolio.transactions:
        print("  No transactions yet.\n")
        return

    print(colored("  ┌─────┬────────────┬──────────────┬────────┬───────────┬────────────┬────────────┐", "dim"))
    print(colored("  │", "dim") + colored(" Day ", "bold") +
          colored("│", "dim") + colored(" Type       ", "bold") +
          colored("│", "dim") + colored(" Ticker       ", "bold") +
          colored("│", "dim") + colored(" Counts ", "bold") +
          colored("│", "dim") + colored("   Price   ", "bold") +
          colored("│", "dim") + colored("   Total    ", "bold") +
          colored("│", "dim") + colored("   Profit   ", "bold") +
          colored("│", "dim"))
    print(colored("  ├─────┼────────────┼──────────────┼────────┼───────────┼────────────┼────────────┤", "dim"))

    for t in portfolio.transactions[-20:]:  # Show last 20
        action_color = "green" if "BUY" in t.action else "red"
        if "EXERCISE" in t.action:
            action_color = "green"
        elif "EXPIRE" in t.action:
            action_color = "dim"
            
        profit_str = ""
        if t.profit != 0:
            profit_str = format_change(t.profit)
            
        print(colored("  │", "dim") +
              f" {t.day:>3} " +
              colored("│", "dim") +
              f" {colored(t.action, action_color):<10} " +
              colored("│", "dim") +
              f" {t.ticker:<12} " +
              colored("│", "dim") +
              f" {t.shares:>6} " +
              colored("│", "dim") +
              f" ${t.price:>8.2f}" +
              colored("│", "dim") +
              f" ${t.total:>9.2f}" +
              colored("│", "dim") +
              f" {profit_str:>10}" +
              colored("│", "dim"))

    print(colored("  └─────┴────────────┴──────────────┴────────┴───────────┴────────────┴────────────┘", "dim"))

    if len(portfolio.transactions) > 20:
        print(f"  (Showing last 20 of {len(portfolio.transactions)} transactions)")
    print()


def print_stock_detail(stock: Stock):
    """Print detailed info about a single stock."""
    history = stock.price_history
    print(colored(f"\n  ═══ {stock.ticker} — {stock.name} ═══", "bold"))
    print(f"  Current Price:  ${stock.price:,.2f}")
    print(f"  Day Change:     {format_change(stock.day_change)} ({format_change(stock.day_change_pct, True)})")
    print(f"  All-Time High:  ${max(history):,.2f}")
    print(f"  All-Time Low:   ${min(history):,.2f}")
    print(f"  Opening Price:  ${history[0]:,.2f}")

    total_change = stock.price - history[0]
    total_change_pct = (total_change / history[0]) * 100 if history[0] != 0 else 0
    print(f"  Total Change:   {format_change(total_change)} ({format_change(total_change_pct, True)})")

    # Simple moving average (last 5 days)
    if len(history) >= 5:
        sma5 = sum(history[-5:]) / 5
        print(f"  SMA (5-day):    ${sma5:,.2f}")

    # Volatility (std dev of daily returns)
    if len(history) >= 3:
        returns = [(history[i] - history[i-1]) / history[i-1] for i in range(1, len(history))]
        avg_ret = sum(returns) / len(returns)
        variance = sum((r - avg_ret) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance) * 100
        print(f"  Daily Vol:      {std_dev:.2f}%")

    print()


# ─────────────────────────────────────────────
# Main Menu & Game Loop
# ─────────────────────────────────────────────
def select_option_params(stock: Stock, day: int) -> Optional[Tuple[str, float, int, float]]:
    """Interactive option selection. Returns (type, strike, expiry_day, premium)."""
    print(f"\n  Option Chain for {stock.ticker} (Price: ${stock.price:.2f})")

    # Simple expiry selection: 10 days from now
    expiry_days = 10
    expiry_day = day + expiry_days
    T = expiry_days / TRADING_DAYS_PER_YEAR

    # Generate strikes around current price
    strikes = []
    base_price = round(stock.price, 0)
    for i in range(-3, 4):
        strike = base_price + (i * 5)  # $5 increments
        if strike > 0:
            strikes.append(strike)

    print(colored(f"  Expiry: {expiry_days} Days (Day {expiry_day})", "bold"))
    print()

    # ── CALLS table ──
    print(colored("  CALLS", "bold"))
    print(colored("  ┌─────┬──────────┬──────────┬────────┬────────┬────────┬────────┐", "dim"))
    print(colored("  │", "dim") + colored(" #   ", "bold") +
          colored("│", "dim") + colored(" Strike   ", "bold") +
          colored("│", "dim") + colored(" Premium  ", "bold") +
          colored("│", "dim") + colored(" Delta  ", "bold") +
          colored("│", "dim") + colored(" Gamma  ", "bold") +
          colored("│", "dim") + colored(" Theta  ", "bold") +
          colored("│", "dim") + colored(" Vega   ", "bold") +
          colored("│", "dim"))
    print(colored("  ├─────┼──────────┼──────────┼────────┼────────┼────────┼────────┤", "dim"))

    option_data = []

    for i, k in enumerate(strikes, 1):
        cp = option_premium(stock, k, T, "call")
        pp = option_premium(stock, k, T, "put")
        cg = option_greeks(stock, k, T, "call")
        pg = option_greeks(stock, k, T, "put")

        option_data.append({
            "strike": k,
            "call": cp, "put": pp,
            "call_greeks": cg, "put_greeks": pg,
        })

        itm = " *" if stock.price > k else "  "
        print(colored("  │", "dim") +
              f" {i:<3} " +
              colored("│", "dim") +
              f" ${k:<7.2f}{itm}" +
              colored("│", "dim") +
              f" ${cp:>7.2f} " +
              colored("│", "dim") +
              f" {cg.delta:>+5.3f} " +
              colored("│", "dim") +
              f" {cg.gamma:>6.4f}" +
              colored("│", "dim") +
              f" {cg.theta:>6.3f}" +
              colored("│", "dim") +
              f" {cg.vega:>6.3f}" +
              colored("│", "dim"))

    print(colored("  └─────┴──────────┴──────────┴────────┴────────┴────────┴────────┘", "dim"))
    print()

    # ── PUTS table ──
    print(colored("  PUTS", "bold"))
    print(colored("  ┌─────┬──────────┬──────────┬────────┬────────┬────────┬────────┐", "dim"))
    print(colored("  │", "dim") + colored(" #   ", "bold") +
          colored("│", "dim") + colored(" Strike   ", "bold") +
          colored("│", "dim") + colored(" Premium  ", "bold") +
          colored("│", "dim") + colored(" Delta  ", "bold") +
          colored("│", "dim") + colored(" Gamma  ", "bold") +
          colored("│", "dim") + colored(" Theta  ", "bold") +
          colored("│", "dim") + colored(" Vega   ", "bold") +
          colored("│", "dim"))
    print(colored("  ├─────┼──────────┼──────────┼────────┼────────┼────────┼────────┤", "dim"))

    for i, d in enumerate(option_data, 1):
        k = d["strike"]
        pp = d["put"]
        pg = d["put_greeks"]
        itm = " *" if stock.price < k else "  "
        print(colored("  │", "dim") +
              f" {i:<3} " +
              colored("│", "dim") +
              f" ${k:<7.2f}{itm}" +
              colored("│", "dim") +
              f" ${pp:>7.2f} " +
              colored("│", "dim") +
              f" {pg.delta:>+5.3f} " +
              colored("│", "dim") +
              f" {pg.gamma:>6.4f}" +
              colored("│", "dim") +
              f" {pg.theta:>6.3f}" +
              colored("│", "dim") +
              f" {pg.vega:>6.3f}" +
              colored("│", "dim"))

    print(colored("  └─────┴──────────┴──────────┴────────┴────────┴────────┴────────┘", "dim"))
    print(colored("  (* = in the money)  Theta = $/day decay  Vega = $/1% vol move", "dim"))

    try:
        idx = int(input("\n  Select Strike (#): ").strip()) - 1
        if 0 <= idx < len(option_data):
            selected = option_data[idx]

            type_choice = input("  Call (C) or Put (P)? ").strip().upper()
            if type_choice in ["C", "CALL"]:
                return "call", selected["strike"], expiry_day, selected["call"]
            elif type_choice in ["P", "PUT"]:
                return "put", selected["strike"], expiry_day, selected["put"]
            else:
                print(colored("  Invalid option type.", "red"))
                return None
        else:
            print(colored("  Invalid selection.", "red"))
            return None
    except ValueError:
        print(colored("  Invalid input.", "red"))
        return None


def print_option_positions(portfolio: Portfolio, stocks: Dict[str, Stock], day: int):
    """Display option positions with selection numbers for exercise/close."""
    if not portfolio.option_holdings:
        print(colored("  You don't hold any option positions.", "red"))
        return

    for i, opt in enumerate(portfolio.option_holdings, 1):
        s = stocks[opt.ticker]
        T = option_time_left(opt.expiry_day, day)
        cur_prem = option_premium(s, opt.strike, T, opt.option_type)
        dte = max(0, opt.expiry_day - day)

        if opt.option_type == "call":
            intrinsic = max(0, s.price - opt.strike)
        else:
            intrinsic = max(0, opt.strike - s.price)

        mkt_val = cur_prem * 100 * opt.contracts
        cost = opt.avg_cost * 100 * opt.contracts
        pnl = mkt_val - cost
        itm_label = colored("ITM", "green") if intrinsic > 0 else colored("OTM", "red")

        print(f"    {i}. {opt.ticker} ${opt.strike:.0f} {opt.option_type.upper()} x{opt.contracts} | "
              f"DTE {dte} | {itm_label} | "
              f"Prem ${cur_prem:.2f} | Intrinsic ${intrinsic:.2f} | "
              f"P&L: {format_change(pnl)}")
    print()


def print_pending_orders(portfolio: Portfolio, stocks: Dict[str, Stock]):
    """Display all pending limit orders."""
    if not portfolio.pending_orders:
        print(colored("  No pending limit orders.", "dim"))
        print()
        return

    print(colored("  ┌─────┬──────┬────────┬────────┬────────────┬────────────┬──────────┐", "dim"))
    print(colored("  │", "dim") + colored("  ID ", "bold") +
          colored("│", "dim") + colored(" Type ", "bold") +
          colored("│", "dim") + colored(" Ticker ", "bold") +
          colored("│", "dim") + colored(" Shares ", "bold") +
          colored("│", "dim") + colored(" Limit $    ", "bold") +
          colored("│", "dim") + colored(" Current $  ", "bold") +
          colored("│", "dim") + colored(" Placed   ", "bold") +
          colored("│", "dim"))
    print(colored("  ├─────┼──────┼────────┼────────┼────────────┼────────────┼──────────┤", "dim"))

    for order in portfolio.pending_orders:
        cur_price = stocks[order.ticker].price if order.ticker in stocks else 0.0
        type_color = "green" if order.order_type == "BUY" else "red"
        print(colored("  │", "dim") +
              f" {order.order_id:>3} " +
              colored("│", "dim") +
              f" {colored(order.order_type, type_color):<4} " +
              colored("│", "dim") +
              f" {order.ticker:<6} " +
              colored("│", "dim") +
              f" {order.shares:>6} " +
              colored("│", "dim") +
              f" ${order.limit_price:>9.2f}" +
              colored("│", "dim") +
              f" ${cur_price:>9.2f}" +
              colored("│", "dim") +
              f" Day {order.day_placed:>3} " +
              colored("│", "dim"))

    print(colored("  └─────┴──────┴────────┴────────┴────────────┴────────────┴──────────┘", "dim"))
    print()


def print_portfolio_analysis(analysis: Dict, day: int):
    """Display comprehensive portfolio analysis."""
    overview = analysis["overview"]
    distribution = analysis["distribution"]
    sectors = analysis["sector_allocation"]
    risk = analysis["risk_metrics"]
    warnings = analysis["warnings"]
    recommendations = analysis["recommendations"]
    news = analysis["news"]
    
    print(colored("\n  ╔══════════════════════════════════════════════════════════════════════╗", "magenta"))
    print(colored("  ║", "magenta") + colored("           📊 PORTFOLIO ANALYSIS REPORT 📊", "bold").center(76) + colored("║", "magenta"))
    print(colored("  ╚══════════════════════════════════════════════════════════════════════╝", "magenta"))
    
    # Portfolio Overview
    print(colored("\n  ┌────────────────────────────────────────────────────────────────────┐", "cyan"))
    print(colored("  │", "cyan") + colored(" PORTFOLIO OVERVIEW", "bold").center(70) + colored("│", "cyan"))
    print(colored("  ├────────────────────────────────────────────────────────────────────┤", "cyan"))
    print(f"  │  Total Value:        {format_money(overview['total_value']):>50}  │")
    print(f"  │  Cash:               {format_money(overview['cash']):>50}  │")
    print(f"  │  Cash %:             {overview['cash_pct']*100:>49.1f}%  │")
    print(f"  │  Stock Value:        {format_money(overview['stock_value']):>50}  │")
    print(f"  │  Option Value:       {format_money(overview['option_value']):>50}  │")
    print(f"  │  Positions:          {overview['num_positions']:>50}  │")
    print(f"  │  Option Positions:   {overview['num_options']:>50}  │")
    print(colored("  ├────────────────────────────────────────────────────────────────────┤", "cyan"))
    print(f"  │  Realized P&L:       {format_money(overview['realized_pnl']):>50}  │")
    print(f"  │  Unrealized P&L:     {format_money(overview['unrealized_pnl']):>50}  │")
    print(f"  │  Total P&L:          {format_money(overview['total_pnl']):>50}  │")
    print(f"  │  ROI:                {format_change(overview['roi_pct'], True):>50}  │")
    print(colored("  └────────────────────────────────────────────────────────────────────┘", "cyan"))
    
    # Risk Metrics
    print(colored("\n  ┌────────────────────────────────────────────────────────────────────┐", "cyan"))
    print(colored("  │", "cyan") + colored(" RISK METRICS", "bold").center(70) + colored("│", "cyan"))
    print(colored("  ├────────────────────────────────────────────────────────────────────┤", "cyan"))
    print(f"  │  Daily Volatility:        {risk['portfolio_volatility']*100:>45.2f}%  │")
    print(f"  │  Annualized Volatility:   {risk['annualized_volatility']*100:>45.2f}%  │")
    print(f"  │  95% VaR (1-day):         {format_money(risk['value_at_risk_95']):>50}  │")
    if risk.get('max_drawdown', 0) < 0:
        print(f"  │  Max Drawdown:            {format_change(risk['max_drawdown']*100, True):>50}  │")
    if risk['current_drawdown'] < 0:
        print(f"  │  Current Drawdown:        {format_change(risk['current_drawdown']*100, True):>50}  │")
    print(colored("  └────────────────────────────────────────────────────────────────────┘", "cyan"))
    
    # Warnings Section
    if warnings:
        print(colored("\n  ⚠️  WARNINGS & ALERTS", "yellow"))
        print(colored("  " + "═" * 70, "yellow"))
        for warning in warnings:
            level_color = {"CRITICAL": "red", "WARNING": "yellow", "INFO": "cyan"}[warning["level"]]
            icon = "🔴" if warning["level"] == "CRITICAL" else ("🟡" if warning["level"] == "WARNING" else "🔵")
            print(f"\n  {icon} {colored(warning['message'], level_color)}")
            print(f"     {colored('Explanation:', 'dim')} {warning['explanation']}")
        print()
    
    # Sector Allocation
    if sectors["sectors"]:
        print(colored("\n  ┌────────────────────────────────────────────────────────────────────┐", "cyan"))
        print(colored("  │", "cyan") + colored(" SECTOR ALLOCATION", "bold").center(70) + colored("│", "cyan"))
        print(colored("  ├────────────────────────────────────────────────────────────────────┤", "cyan"))
        for sector_name, data in sorted(sectors["sectors"].items(), key=lambda x: x[1]["value"], reverse=True):
            pct = data["pct"] * 100
            bar_len = int(pct / 2)
            bar = "█" * bar_len + "░" * (25 - bar_len)
            risk_level = data.get("risk", "medium")
            risk_icon = {"low": "🟢", "low-medium": "🟢", "medium": "🟡", "medium-high": "🟠", "high": "🔴"}.get(risk_level, "🟡")
            print(f"  │  {sector_name:<25} {bar} {pct:>5.1f}%  {risk_icon}  │")
        print(colored("  └────────────────────────────────────────────────────────────────────┘", "cyan"))
    
    # Position Distribution
    if distribution["positions"]:
        print(colored("\n  ┌────────────────────────────────────────────────────────────────────────────────────┐", "cyan"))
        print(colored("  │", "cyan") + colored(" POSITION DISTRIBUTION", "bold").center(86) + colored("│", "cyan"))
        print(colored("  ├────────────────────────────────────────────────────────────────────────────────────┤", "cyan"))
        print(colored("  │", "cyan") + " Ticker          Value        %Port    P&L        Sector".ljust(86) + colored("│", "cyan"))
        print(colored("  ├────────────────────────────────────────────────────────────────────────────────────┤", "cyan"))
        for pos in distribution["positions"][:10]:  # Top 10 positions
            ticker = pos["ticker"][:15]
            value = pos["value"]
            pct = pos["pct"] * 100
            pnl = pos.get("pnl", 0)
            pnl_str = format_change(pnl)
            sector = pos.get("sector", "Unknown")[:15]
            print(f"  │  {ticker:<15} ${value:>10,.2f}  {pct:>6.1f}%  {pnl_str:>12}  {sector:<15}  │")
        print(colored("  └────────────────────────────────────────────────────────────────────────────────────┘", "cyan"))
    
    # Position Analysis (Pros/Cons)
    if analysis["position_analysis"]:
        print(colored("\n  📋 POSITION ANALYSIS", "yellow"))
        print(colored("  " + "═" * 70, "yellow"))
        for pos in analysis["position_analysis"]:
            if pos["type"] == "stock":
                print(f"\n  📈 {colored(pos['ticker'], 'bold')} - {pos['name']}")
                print(f"     Shares: {pos['shares']:,} | Avg Cost: ${pos['avg_cost']:.2f} | Current: ${pos['current_price']:.2f}")
                print(f"     P&L: {format_change(pos['pnl'])} ({format_change(pos['pnl_pct'], True)})")
            else:
                print(f"\n  📉 {colored(pos['ticker'] + ' ' + pos['option_type'].upper(), 'bold')} Strike ${pos['strike']:.0f}")
                print(f"     Contracts: {pos['contracts']} | DTE: {pos['dte']} | Moneyness: {pos['moneyness']}")
                print(f"     P&L: {format_change(pos['pnl'])} ({format_change(pos['pnl_pct'], True)})")
            
            if pos["pros"]:
                print(f"     {colored('✅ Pros:', 'green')}")
                for pro in pos["pros"]:
                    print(f"        • {pro}")
            if pos["cons"]:
                print(f"     {colored('❌ Cons:', 'red')}")
                for con in pos["cons"]:
                    print(f"        • {con}")
            print(f"     {colored('💡 Recommendation:', 'cyan')} {pos['recommendation']}")
    
    # Recommendations
    if recommendations:
        print(colored("\n  💡 PORTFOLIO RECOMMENDATIONS", "green"))
        print(colored("  " + "═" * 70, "green"))
        for rec in recommendations:
            priority_color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "cyan"}.get(rec.get("priority", "MEDIUM"), "yellow")
            priority_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}.get(rec.get("priority", "MEDIUM"), "🟡")
            print(f"\n  {priority_icon} {colored(rec['title'], 'bold')} [{colored(rec.get('priority', 'MEDIUM'), priority_color)}]")
            print(f"     {rec['action']}")
    
    # Market News
    if news:
        print(colored("\n  📰 MARKET NEWS", "cyan"))
        print(colored("  " + "═" * 70, "cyan"))
        for item in news:
            sentiment_icon = {"positive": "📈", "negative": "📉", "neutral": "📰"}[item["sentiment"]]
            sentiment_color = {"positive": "green", "negative": "red", "neutral": "dim"}[item["sentiment"]]
            print(f"  {sentiment_icon} {colored(item['headline'], sentiment_color)}")
    
    print()


def print_menu():
    print(colored("  ┌──────────────────────────────────────┐", "cyan"))
    print(colored("  │", "cyan") + colored("           ACTIONS MENU", "bold") + colored("               │", "cyan"))
    print(colored("  ├──────────────────────────────────────┤", "cyan"))
    print(colored("  │", "cyan") + "  [1] View Market Overview            " + colored("│", "cyan"))
    print(colored("  │", "cyan") + "  [2] View Stock Chart                " + colored("│", "cyan"))
    print(colored("  │", "cyan") + "  [3] View Stock Details              " + colored("│", "cyan"))
    print(colored("  │", "cyan") + "  [4] Buy Stock                       " + colored("│", "cyan"))
    print(colored("  │", "cyan") + "  [5] Sell Stock                      " + colored("│", "cyan"))
    print(colored("  │", "cyan") + "  [6] Buy Option                      " + colored("│", "cyan"))
    print(colored("  │", "cyan") + "  [7] Exercise Option (early)         " + colored("│", "cyan"))
    print(colored("  │", "cyan") + "  [8] Close Option Position           " + colored("│", "cyan"))
    print(colored("  │", "cyan") + "  [9] Place Limit Order               " + colored("│", "cyan"))
    print(colored("  │", "cyan") + "  [0] View / Cancel Limit Orders      " + colored("│", "cyan"))
    print(colored("  │", "cyan") + "  [P] View Portfolio                  " + colored("│", "cyan"))
    print(colored("  │", "cyan") + "  [A] Portfolio Analysis              " + colored("│", "cyan"))
    print(colored("  │", "cyan") + "  [T] Transaction History             " + colored("│", "cyan"))
    print(colored("  │", "cyan") + "  [N] Next Day                        " + colored("│", "cyan"))
    print(colored("  │", "cyan") + "  [S] Skip Multiple Days              " + colored("│", "cyan"))
    print(colored("  │", "cyan") + "  [R] Restart Simulation              " + colored("│", "cyan"))
    print(colored("  │", "cyan") + "  [Q] Quit                            " + colored("│", "cyan"))
    print(colored("  └──────────────────────────────────────┘", "cyan"))


def select_stock(stocks: Dict[str, Stock], prompt: str = "Select stock (1-10 or ticker): ") -> Optional[Stock]:
    """Let user select a stock by number or ticker."""
    tickers = list(stocks.keys())
    for i, t in enumerate(tickers, 1):
        s = stocks[t]
        print(f"    {i:>2}. {t:<6} ${s.price:>8.2f}  {format_change(s.day_change)} ({format_change(s.day_change_pct, True)})")
    print()

    choice = input(f"  {prompt}").strip().upper()
    if not choice:
        return None

    # Try as number
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(tickers):
            return stocks[tickers[idx]]
    except ValueError:
        pass

    # Try as ticker
    if choice in stocks:
        return stocks[choice]

    print(colored("  Invalid selection.", "red"))
    return None


def get_positive_int(prompt: str) -> Optional[int]:
    """Get a positive integer from user input."""
    try:
        val = int(input(f"  {prompt}").strip())
        if val <= 0:
            print(colored("  Please enter a positive number.", "red"))
            return None
        return val
    except (ValueError, EOFError):
        print(colored("  Invalid number.", "red"))
        return None


def advance_day(stocks: Dict[str, Stock], portfolio: Portfolio, day: int):
    """Simulate one day of market activity."""
    # Add some market-wide correlation
    market_sentiment = random.gauss(0, 0.005)  # slight market-wide push
    for stock in stocks.values():
        # Temporarily adjust drift with market sentiment
        original_drift = stock.drift
        stock.drift += market_sentiment
        stock.simulate_day()
        stock.drift = original_drift
    
    # Check for option expiry
    messages = portfolio.check_option_expiry(stocks, day)

    # Check limit orders after prices have updated
    limit_msgs = portfolio.check_limit_orders(stocks, day)
    messages.extend(limit_msgs)

    portfolio.record_value(day, stocks)

    if messages:
        print(colored("\n  🔔 ALERTS:", "yellow"))
        for msg in messages:
            print(f"  {msg}")
        input("\n  Press Enter to continue...")


def end_of_simulation(portfolio: Portfolio, stocks: Dict[str, Stock], day: int):
    """Display final results."""
    clear_screen()
    total_value = portfolio.get_portfolio_value(stocks, day)
    total_pnl = total_value - portfolio.starting_balance
    total_pnl_pct = (total_pnl / portfolio.starting_balance) * 100

    print(colored("\n  ╔══════════════════════════════════════════════════════════╗", "yellow"))
    print(colored("  ║", "yellow") + colored("         🏁  SIMULATION COMPLETE!  🏁", "bold").center(67) + colored("║", "yellow"))
    print(colored("  ╚══════════════════════════════════════════════════════════╝", "yellow"))
    print()
    print(f"  Days Simulated:       {day}")
    print(f"  Starting Balance:     ${portfolio.starting_balance:>12,.2f}")
    print(f"  Final Portfolio Value: ${total_value:>12,.2f}")
    print(f"  {'─' * 45}")
    print(f"  Total P&L:            {format_money(total_pnl)} ({format_change(total_pnl_pct, True)})")
    print(f"  Realized P&L:         {format_money(portfolio.realized_pnl)}")
    print(f"  Unrealized P&L:       {format_money(portfolio.get_unrealized_pnl(stocks, day))}")
    print(f"  {'─' * 45}")
    print(f"  Total Transactions:   {len(portfolio.transactions)}")
    print(f"  Cash Remaining:       ${portfolio.cash:>12,.2f}")

    # Best and worst holdings
    if portfolio.holdings:
        best = max(portfolio.holdings.values(), key=lambda h: h.shares * (stocks[h.ticker].price - h.avg_cost))
        worst = min(portfolio.holdings.values(), key=lambda h: h.shares * (stocks[h.ticker].price - h.avg_cost))
        best_pnl = best.shares * (stocks[best.ticker].price - best.avg_cost)
        worst_pnl = worst.shares * (stocks[worst.ticker].price - worst.avg_cost)
        print(f"  Best Holding:         {best.ticker} ({format_money(best_pnl)})")
        print(f"  Worst Holding:        {worst.ticker} ({format_money(worst_pnl)})")

    # Best and worst performing stocks overall
    best_stock = max(stocks.values(), key=lambda s: (s.price - s.price_history[0]) / s.price_history[0])
    worst_stock = min(stocks.values(), key=lambda s: (s.price - s.price_history[0]) / s.price_history[0])
    best_ret = (best_stock.price - best_stock.price_history[0]) / best_stock.price_history[0] * 100
    worst_ret = (worst_stock.price - worst_stock.price_history[0]) / worst_stock.price_history[0] * 100
    print(f"  {'─' * 45}")
    print(f"  Best Market Stock:    {best_stock.ticker} ({format_change(best_ret, True)})")
    print(f"  Worst Market Stock:   {worst_stock.ticker} ({format_change(worst_ret, True)})")
    print()

    if total_pnl > 0:
        print(colored("  🎉 Congratulations! You made a profit!", "green"))
    elif total_pnl < 0:
        print(colored("  📉 Better luck next time! You had a loss.", "red"))
    else:
        print(colored("  ⚖️  You broke even!", "yellow"))
    print()


def init_stocks() -> Dict[str, Stock]:
    """Initialize all stocks."""
    stocks = {}
    for ticker, name, price, vol, drift, sector, industry in STOCK_DEFINITIONS:
        stocks[ticker] = Stock(ticker=ticker, name=name, price=price, volatility=vol, drift=drift, sector=sector, industry=industry)
    return stocks


def run_simulation():
    """Main simulation loop."""
    stocks = init_stocks()
    portfolio = Portfolio(STARTING_BALANCE)
    day = 1

    PortfolioAnalyzer.SHARED_NEWS.reset()
    portfolio.record_value(day, stocks)

    clear_screen()
    print(colored("\n  ╔══════════════════════════════════════════════════════════╗", "cyan"))
    print(colored("  ║", "cyan") + colored("       Welcome to the Stock Market Simulator!", "bold").center(67) + colored("║", "cyan"))
    print(colored("  ╠══════════════════════════════════════════════════════════╣", "cyan"))
    print(colored("  ║", "cyan") + f"  You start with ${STARTING_BALANCE:,.2f}. Trade wisely over {MAX_DAYS} days!   " + colored("║", "cyan"))
    print(colored("  ╚══════════════════════════════════════════════════════════╝\n", "cyan"))

    input("  Press Enter to begin...")

    while day <= MAX_DAYS:
        clear_screen()
        print_header(day, portfolio, stocks)
        print_market_table(stocks, day)
        print_menu()

        choice = input("\n  Enter choice: ").strip().upper()

        if choice == "1":
            clear_screen()
            print_header(day, portfolio, stocks)
            print_market_table(stocks, day)
            input("  Press Enter to continue...")

        elif choice == "2":
            print("\n  Select a stock to view its chart:")
            stock = select_stock(stocks)
            if stock:
                clear_screen()
                print(render_chart(stock))
                input("  Press Enter to continue...")

        elif choice == "3":
            print("\n  Select a stock for details:")
            stock = select_stock(stocks)
            if stock:
                clear_screen()
                print_stock_detail(stock)
                print(render_chart(stock))
                input("  Press Enter to continue...")

        elif choice == "4":
            print(f"\n  💰 Cash available: ${portfolio.cash:,.2f}")
            print("\n  Select a stock to buy:")
            stock = select_stock(stocks)
            if stock:
                max_shares = int(portfolio.cash / stock.price)
                print(f"  {stock.ticker} @ ${stock.price:,.2f} — You can buy up to {max_shares} shares")
                shares = get_positive_int(f"How many shares to buy? (max {max_shares}): ")
                if shares:
                    success, msg = portfolio.buy(stock, shares, day)
                    print(f"\n  {colored(msg, 'green' if success else 'red')}")
                    input("  Press Enter to continue...")

        elif choice == "5":
            if not portfolio.holdings:
                print(colored("\n  You don't own any stocks to sell.", "red"))
                input("  Press Enter to continue...")
            else:
                print("\n  Select a stock to sell:")
                owned_tickers = list(portfolio.holdings.keys())
                for i, t in enumerate(owned_tickers, 1):
                    h = portfolio.holdings[t]
                    cur_price = stocks[t].price
                    pnl = h.shares * (cur_price - h.avg_cost)
                    print(f"    {i}. {t:<6} | {h.shares} shares @ avg ${h.avg_cost:,.2f} | Now: ${cur_price:,.2f} | P&L: {format_change(pnl)}")
                print()
                sel = input("  Select (number or ticker): ").strip().upper()

                selected_ticker = None
                try:
                    idx = int(sel) - 1
                    if 0 <= idx < len(owned_tickers):
                        selected_ticker = owned_tickers[idx]
                except ValueError:
                    if sel in portfolio.holdings:
                        selected_ticker = sel

                if selected_ticker:
                    h = portfolio.holdings[selected_ticker]
                    shares = get_positive_int(f"How many shares to sell? (own {h.shares}): ")
                    if shares:
                        success, msg = portfolio.sell(stocks[selected_ticker], shares, day)
                        print(f"\n  {colored(msg, 'green' if success else 'red')}")
                        input("  Press Enter to continue...")
                else:
                    print(colored("  Invalid selection.", "red"))
                    input("  Press Enter to continue...")

        elif choice == "6":
            print(f"\n  💰 Cash available: ${portfolio.cash:,.2f}")
            print("\n  Select a stock for options:")
            stock = select_stock(stocks)
            if stock:
                params = select_option_params(stock, day)
                if params:
                    opt_type, strike, expiry, premium = params
                    print(f"\n  Selected: {stock.ticker} ${strike} {opt_type.upper()} Expiring Day {expiry}")
                    print(f"  Premium: ${premium:.2f} per share (${premium * 100:.2f} per contract)")

                    max_contracts = int(portfolio.cash / (premium * 100))
                    contracts = get_positive_int(f"How many contracts? (max {max_contracts}): ")

                    if contracts:
                        success, msg = portfolio.buy_option(stock, opt_type, strike, expiry, contracts, premium, day)
                        print(f"\n  {colored(msg, 'green' if success else 'red')}")
                        input("  Press Enter to continue...")

        elif choice == "7":
            # Exercise option early
            if not portfolio.option_holdings:
                print(colored("\n  You don't hold any option positions.", "red"))
                input("  Press Enter to continue...")
            else:
                print(colored("\n  ═══ EXERCISE OPTION (Early) ═══", "bold"))
                print("  Only ITM options can be exercised.\n")
                print_option_positions(portfolio, stocks, day)
                try:
                    sel = int(input("  Select option (#): ").strip()) - 1
                    if 0 <= sel < len(portfolio.option_holdings):
                        opt = portfolio.option_holdings[sel]
                        if opt.option_type == "call":
                            intrinsic = stocks[opt.ticker].price - opt.strike
                        else:
                            intrinsic = opt.strike - stocks[opt.ticker].price
                        if intrinsic <= 0:
                            print(colored(f"  Option is OTM (intrinsic ${intrinsic:.2f}). Cannot exercise.", "red"))
                        else:
                            payout = intrinsic * 100 * opt.contracts
                            confirm = input(f"  Exercise for ${payout:,.2f} payout? (y/n): ").strip().lower()
                            if confirm == "y":
                                success, msg = portfolio.exercise_option(sel, stocks, day)
                                print(f"\n  {colored(msg, 'green' if success else 'red')}")
                    else:
                        print(colored("  Invalid selection.", "red"))
                except ValueError:
                    print(colored("  Invalid input.", "red"))
                input("  Press Enter to continue...")

        elif choice == "8":
            # Close option position
            if not portfolio.option_holdings:
                print(colored("\n  You don't hold any option positions.", "red"))
                input("  Press Enter to continue...")
            else:
                print(colored("\n  ═══ CLOSE OPTION POSITION ═══", "bold"))
                print("  Sell back at current market premium.\n")
                print_option_positions(portfolio, stocks, day)
                try:
                    sel = int(input("  Select option (#): ").strip()) - 1
                    if 0 <= sel < len(portfolio.option_holdings):
                        opt = portfolio.option_holdings[sel]
                        s = stocks[opt.ticker]
                        T = option_time_left(opt.expiry_day, day)
                        cur = option_premium(s, opt.strike, T, opt.option_type)
                        revenue = cur * 100 * opt.contracts
                        cost = opt.avg_cost * 100 * opt.contracts
                        pnl = revenue - cost
                        print(f"  Current premium: ${cur:.2f}/sh  →  Revenue: ${revenue:,.2f}  (P&L: {format_change(pnl)})")
                        confirm = input(f"  Close this position? (y/n): ").strip().lower()
                        if confirm == "y":
                            success, msg = portfolio.close_option(sel, stocks, day)
                            print(f"\n  {colored(msg, 'green' if success else 'red')}")
                    else:
                        print(colored("  Invalid selection.", "red"))
                except ValueError:
                    print(colored("  Invalid input.", "red"))
                input("  Press Enter to continue...")

        elif choice == "9":
            # Place limit order
            print(colored("\n  ═══ PLACE LIMIT ORDER ═══", "bold"))
            print(f"  💰 Cash available: ${portfolio.cash:,.2f}")
            type_choice = input("  Buy or Sell? (B/S): ").strip().upper()
            if type_choice in ("B", "BUY"):
                order_type = "BUY"
            elif type_choice in ("S", "SELL"):
                order_type = "SELL"
            else:
                print(colored("  Invalid order type.", "red"))
                input("  Press Enter to continue...")
                continue

            if order_type == "SELL" and not portfolio.holdings:
                print(colored("  You don't own any stocks to sell.", "red"))
                input("  Press Enter to continue...")
                continue

            print(f"\n  Select a stock for LIMIT {order_type}:")
            stock = select_stock(stocks)
            if stock:
                try:
                    limit_price = float(input(f"  Enter limit price (current ${stock.price:,.2f}): ").strip())
                except (ValueError, EOFError):
                    print(colored("  Invalid price.", "red"))
                    input("  Press Enter to continue...")
                    continue

                if order_type == "BUY":
                    max_shares = int(portfolio.cash / limit_price) if limit_price > 0 else 0
                    shares = get_positive_int(f"How many shares? (max {max_shares}): ")
                else:
                    if stock.ticker in portfolio.holdings:
                        reserved = sum(o.shares for o in portfolio.pending_orders
                                       if o.ticker == stock.ticker and o.order_type == "SELL")
                        avail = portfolio.holdings[stock.ticker].shares - reserved
                    else:
                        avail = 0
                    shares = get_positive_int(f"How many shares? (available {avail}): ")

                if shares:
                    success, msg = portfolio.place_limit_order(stock.ticker, order_type, shares, limit_price, day)
                    print(f"\n  {colored(msg, 'green' if success else 'red')}")
                input("  Press Enter to continue...")

        elif choice == "0":
            # View / Cancel limit orders
            clear_screen()
            print(colored("\n  ═══ PENDING LIMIT ORDERS ═══", "bold"))
            print_pending_orders(portfolio, stocks)
            if portfolio.pending_orders:
                cancel = input("  Cancel an order? Enter order ID (or Enter to go back): ").strip()
                if cancel:
                    try:
                        oid = int(cancel)
                        success, msg = portfolio.cancel_limit_order(oid)
                        print(f"\n  {colored(msg, 'green' if success else 'red')}")
                    except ValueError:
                        print(colored("  Invalid ID.", "red"))
            input("  Press Enter to continue...")

        elif choice == "P":
            clear_screen()
            print_header(day, portfolio, stocks)
            print_portfolio_detail(portfolio, stocks, day)
            input("  Press Enter to continue...")

        elif choice == "A":
            clear_screen()
            print_header(day, portfolio, stocks)
            analyzer = PortfolioAnalyzer(portfolio, stocks)
            analysis = analyzer.analyze(day)
            print_portfolio_analysis(analysis, day)
            input("  Press Enter to continue...")

        elif choice == "T":
            clear_screen()
            print_transaction_history(portfolio)
            input("  Press Enter to continue...")

        elif choice == "N":
            if day >= MAX_DAYS:
                print(colored(f"\n  You've reached day {MAX_DAYS}. The simulation is over!", "yellow"))
                input("  Press Enter to see final results...")
                break
            day += 1
            advance_day(stocks, portfolio, day)
            print(colored(f"\n  ⏭  Advanced to Day {day}", "yellow"))

        elif choice == "S":
            remaining = MAX_DAYS - day
            if remaining <= 0:
                print(colored(f"\n  You've reached day {MAX_DAYS}. The simulation is over!", "yellow"))
                input("  Press Enter to see final results...")
                break
            skip = get_positive_int(f"How many days to skip? (1-{remaining}): ")
            if skip:
                skip = min(skip, remaining)
                print(f"\n  Simulating {skip} days...")
                for _ in range(skip):
                    day += 1
                    advance_day(stocks, portfolio, day)
                print(colored(f"  ⏭  Advanced to Day {day}", "yellow"))
                print("\n  Market after skip:")
                for ticker, stock in stocks.items():
                    print(f"    {ticker:<6} ${stock.price:>8.2f}  {format_change(stock.day_change)} ({format_change(stock.day_change_pct, True)})")
                print()
                input("  Press Enter to continue...")

        elif choice == "R":
            confirm = input("  Are you sure you want to restart? (y/n): ").strip().lower()
            if confirm == "y":
                return True

        elif choice == "Q":
            confirm = input("  Are you sure you want to quit? (y/n): ").strip().lower()
            if confirm == "y":
                end_of_simulation(portfolio, stocks, day)
                return False

        else:
            print(colored("  Invalid choice. Please try again.", "red"))
            input("  Press Enter to continue...")

    # End of simulation (day > MAX_DAYS)
    end_of_simulation(portfolio, stocks, day)

    while True:
        choice = input("  [R] Restart  [Q] Quit: ").strip().upper()
        if choice == "R":
            return True
        elif choice == "Q":
            return False


def main():
    """Entry point."""
    print(colored("\n  Loading Stock Market Simulator...\n", "cyan"))
    restart = True
    while restart:
        restart = run_simulation()
    print(colored("  Thanks for playing! Goodbye. 👋\n", "cyan"))


if __name__ == "__main__":
    main()
