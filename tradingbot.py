from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta
from finbert_utils import estimate_sentiment

# Alpaca API credentials and base URL for paper trading
API_KEY = ""
API_SECRET = ""
BASE_URL = ""

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}


class MLTrader(Strategy):
    """
    A Lumibot strategy that uses FinBERT sentiment analysis
    to drive buy and sell decisions for a given symbol.
    """

    def initialize(self, symbol: str = "SPY", cash_at_risk: float = .5):
        """
        Strategy initialization.

        Parameters:
        symbol (str):   Trading symbol, default is SPY.
        cash_at_risk (float): Proportion of total cash to risk per trade.
        """
        # Set the trading symbol and risk parameters
        self.symbol = symbol
        self.cash_at_risk = cash_at_risk

        # Time between sentiment checks/trades
        self.sleeptime = "24H"
        self.last_trade = None  # Track last action to avoid repeat trades

        # Alpaca REST API client for live/paper trading
        self.api = REST(
            base_url=BASE_URL, 
            key_id=API_KEY, 
            secret_key=API_SECRET
        )

    def position_sizing(self):
        """
        Determine buy/sell quantity based on available cash and price.

        Returns:
        cash (float)      : Current account cash balance.
        last_price (float): Latest market price of the symbol.
        quantity (int)    : Number of shares to trade.
        """
        cash = self.get_cash()  # Fetch current cash balance
        last_price = self.get_last_price(self.symbol)  # Current symbol price
        # Compute how many shares we can buy/sell with designated risk
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def get_dates(self):
        """
        Generate string dates for the sentiment lookback period.

        Returns:
        today (str)            : Today's date in YYYY-MM-DD format.
        three_days_prior (str) : Date three days before today.
        """
        today = self.get_datetime()  # Current simulation/live time
        three_days_prior = today - Timedelta(days=3)
        return (
            today.strftime("%Y-%m-%d"),
            three_days_prior.strftime("%Y-%m-%d")
        )

    def get_sentiment(self):
        """
        Pull news headlines from Alpaca and estimate sentiment
        using FinBERT.

        Returns:
        probability (float): Model confidence score.
        sentiment (str)    : "positive" or "negative".
        """
        # Fetch date range strings
        today, three_days_prior = self.get_dates()
        # Retrieve news articles for the symbol
        news = self.api.get_news(
            symbol=self.symbol,
            start=three_days_prior,
            end=today
        )
        # Extract raw headline text from each article
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        # Use FinBERT to estimate sentiment and probability
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment

    def on_trading_iteration(self):
        """
        Core logic executed on each trading cycle.
        Checks sentiment and places bracket orders accordingly.
        """
        # Determine available funds and trade size
        cash, last_price, quantity = self.position_sizing()
        # Get the latest sentiment signal
        probability, sentiment = self.get_sentiment()

        # Only trade if we have enough cash for at least one share
        if cash > last_price:
            # Bullish signal: high-confidence positive sentiment
            if sentiment == "positive" and probability > .999:
                # Avoid flipping position repeatedly
                if self.last_trade == "sell":
                    self.sell_all()
                # Create bracket order: entry + take-profit + stop-loss
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.20,
                    stop_loss_price=last_price * 0.95
                )
                self.submit_order(order)
                self.last_trade = "buy"

            # Bearish signal: high-confidence negative sentiment
            elif sentiment == "negative" and probability > .999:
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * 0.80,
                    stop_loss_price=last_price * 1.05
                )
                self.submit_order(order)
                self.last_trade = "sell"


# Backtesting configuration using Yahoo Finance data
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 12, 31)

# Instantiate the Alpaca broker for paper trading
broker = Alpaca(ALPACA_CREDS)

# Set up the strategy with name, broker, and parameters
strategy = MLTrader(
    name='mlstrat',
    broker=broker,
    parameters={
        "symbol": "SPY",
        "cash_at_risk": .5
    }
)

# Execute backtest over the defined date range
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={
        "symbol": "SPY",
        "cash_at_risk": .5
    }
)
