import os
import time
import json
import math
import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# Third-party broker/data
try:
    import alpaca_trade_api as tradeapi
except Exception:  # pragma: no cover
    tradeapi = None

warnings.filterwarnings('ignore')

# Load env (allow default path and configurable override)
ENV_PATH = os.environ.get('ENV_PATH', '/workspace/config/.env')
load_dotenv(ENV_PATH)

# Structured logging setup
os.makedirs('/workspace/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[
        logging.FileHandler('/workspace/logs/advanced_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('phd_trading_bot')


@dataclass
class Signal:
    action: str
    confidence: float
    reason: str
    market_regime: str
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None


class MarketDataProvider:
    """Aggregates market, fundamentals, and sentiment."""

    def __init__(self) -> None:
        self.alpha_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
        self.news_key = os.getenv('NEWS_API_KEY', '')

    def get_fundamental_data(self, symbol: str) -> Dict:
        if not self.alpha_key:
            return {}
        try:
            url = (
                'https://www.alphavantage.co/query?function=OVERVIEW'
                f'&symbol={symbol}&apikey={self.alpha_key}'
            )
            data = requests.get(url, timeout=10).json()
            if 'Symbol' not in data:
                return {}
            def fget(field: str, default: float = 0.0) -> float:
                value = data.get(field)
                if value in (None, 'None', ''):
                    return default
                try:
                    return float(value)
                except Exception:
                    return default
            return {
                'pe_ratio': fget('PERatio'),
                'peg_ratio': fget('PEGRatio'),
                'price_to_book': fget('PriceToBookRatio'),
                'dividend_yield': fget('DividendYield'),
                'eps': fget('EPS'),
                'market_cap': fget('MarketCapitalization'),
                'revenue_per_share': fget('RevenuePerShareTTM'),
                'profit_margin': fget('ProfitMargin'),
            }
        except Exception as exc:
            logger.warning(f'Fundamentals fetch failed for {symbol}: {exc}')
            return {}

    def get_news_sentiment(self, symbol: str) -> float:
        if not self.news_key:
            return 0.0
        try:
            url = (
                'https://newsapi.org/v2/everything'
                f'?q={symbol}&apiKey={self.news_key}&sortBy=publishedAt&pageSize=10'
            )
            response = requests.get(url, timeout=10).json()
            articles = response.get('articles', [])[:6]
            pos_words = ['up', 'gain', 'profit', 'growth', 'surge', 'bull', 'strong', 'beat', 'exceed']
            neg_words = ['down', 'loss', 'decline', 'fall', 'bear', 'weak', 'miss', 'concern', 'drop']
            score = 0
            for a in articles:
                text = f"{a.get('title','').lower()} {a.get('description','').lower()}"
                score += sum(w in text for w in pos_words)
                score -= sum(w in text for w in neg_words)
            return float(np.clip(score / 10.0, -1.0, 1.0))
        except Exception as exc:
            logger.warning(f'News sentiment failed for {symbol}: {exc}')
            return 0.0


class AdvancedIndicators:
    """Vectorized technicals and features."""

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if len(data) < 210:
            return data
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        data['sma_20'] = close.rolling(20).mean()
        data['sma_50'] = close.rolling(50).mean()
        data['sma_200'] = close.rolling(200).mean()
        data['ema_12'] = close.ewm(span=12, adjust=False).mean()
        data['ema_26'] = close.ewm(span=26, adjust=False).mean()
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_hist'] = data['macd'] - data['macd_signal']

        def rsi(series: pd.Series, period: int) -> pd.Series:
            delta = series.diff()
            gain = delta.where(delta > 0, 0.0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            out = 100 - (100 / (1 + rs))
            return out.fillna(50.0)
        data['rsi_14'] = rsi(close, 14)
        data['rsi_21'] = rsi(close, 21)

        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        data['stoch_k'] = 100 * ((close - low_14) / (high_14 - low_14)).clip(0, 1)
        data['stoch_d'] = data['stoch_k'].rolling(3).mean()

        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        data['bb_upper_2'] = bb_mid + 2 * bb_std
        data['bb_lower_2'] = bb_mid - 2 * bb_std
        data['bb_width'] = (data['bb_upper_2'] - data['bb_lower_2']) / bb_mid.replace(0, np.nan)

        data['high_low'] = high - low
        data['high_close'] = (high - close.shift()).abs()
        data['low_close'] = (low - close.shift()).abs()
        data['true_range'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)
        data['atr_14'] = data['true_range'].rolling(14).mean()

        typical = (high + low + close) / 3
        money_flow = typical * volume
        pos_flow = money_flow.where(typical > typical.shift(), 0).rolling(14).sum()
        neg_flow = money_flow.where(typical < typical.shift(), 0).rolling(14).sum()
        data['mfi'] = 100 - (100 / (1 + (pos_flow / neg_flow.replace(0, np.nan))))

        data['williams_r'] = -100 * ((high_14 - close) / (high_14 - low_14).replace(0, np.nan))
        data['cci'] = (typical - typical.rolling(20).mean()) / (0.015 * typical.rolling(20).std())

        data['vol_sma'] = volume.rolling(20).mean()
        data['vol_ratio'] = (volume / data['vol_sma']).replace([np.inf, -np.inf], np.nan)

        data['mom_5'] = close.pct_change(5)
        data['mom_10'] = close.pct_change(10)
        return data


class MarketRegimeDetector:
    """Detect BULL/BEAR/NORMAL/VOLATILE based on SPY features."""

    def detect(self, spy_df: pd.DataFrame) -> str:
        if spy_df is None or len(spy_df) < 200:
            return 'NORMAL'
        data = AdvancedIndicators.add_indicators(spy_df)
        cur = data.iloc[-1]
        recent_vol = data['atr_14'].tail(20).mean()
        long_vol = data['atr_14'].tail(100).mean()
        if pd.isna(recent_vol) or pd.isna(long_vol):
            return 'NORMAL'
        if cur['sma_20'] > cur['sma_50'] > cur['sma_200'] and cur['close'] > cur['sma_20']:
            return 'BULL' if recent_vol < long_vol * 0.9 else 'VOLATILE'
        if cur['sma_20'] < cur['sma_50'] < cur['sma_200'] and cur['close'] < cur['sma_20']:
            return 'BEAR'
        return 'VOLATILE' if recent_vol > long_vol * 1.2 else 'NORMAL'


class AdvancedStrategy:
    """Multi-factor scoring with regime, fundamentals, and sentiment."""

    def __init__(self) -> None:
        self.data_provider = MarketDataProvider()
        self.regime_detector = MarketRegimeDetector()
        self.min_confidence = 0.65
        self.market_regime = 'NORMAL'

    def _fundamental_score(self, symbol: str) -> float:
        f = self.data_provider.get_fundamental_data(symbol)
        if not f:
            return 0.0
        score = 0.0
        max_s = 0.0
        pe = f.get('pe_ratio', 0.0)
        if 10 <= pe <= 25:
            score += 0.2
        elif 5 <= pe < 10 or 25 < pe <= 35:
            score += 0.1
        max_s += 0.2
        peg = f.get('peg_ratio', 0.0)
        if 0 < peg <= 1:
            score += 0.2
        elif 1 < peg <= 2:
            score += 0.1
        max_s += 0.2
        pm = f.get('profit_margin', 0.0)
        if pm > 0.15:
            score += 0.15
        elif pm > 0.05:
            score += 0.1
        max_s += 0.15
        dy = f.get('dividend_yield', 0.0)
        if 0.02 <= dy <= 0.06:
            score += 0.1
        max_s += 0.1
        return score / max_s if max_s > 0 else 0.0

    def generate_signal(self, df: pd.DataFrame, symbol: str, spy_df: Optional[pd.DataFrame]) -> Signal:
        data = AdvancedIndicators.add_indicators(df)
        if len(data) < 200:
            return Signal('hold', 0.0, 'Insufficient data', self.market_regime)

        if spy_df is not None and len(spy_df) >= 200:
            self.market_regime = self.regime_detector.detect(spy_df)

        cur = data.iloc[-1]
        prev = data.iloc[-2]

        bull: List[Tuple[str, float]] = []
        bear: List[Tuple[str, float]] = []

        # Trend (25%)
        trend_s = 0
        if cur['close'] > cur['sma_20'] > cur['sma_50']:
            trend_s += 2
        if cur['sma_20'] > cur['sma_50'] > cur['sma_200']:
            trend_s += 3
        if cur['ema_12'] > cur['ema_26']:
            trend_s += 1
        if trend_s >= 4:
            bull.append(('Strong Uptrend', 0.25))
        elif trend_s >= 2:
            bull.append(('Moderate Uptrend', 0.15))
        elif trend_s <= -4:
            bear.append(('Strong Downtrend', 0.25))
        elif trend_s <= -2:
            bear.append(('Moderate Downtrend', 0.15))

        # Momentum (20%)
        mom_s = 0
        if cur['macd'] > cur['macd_signal'] and prev['macd'] <= prev['macd_signal']:
            mom_s += 2
        elif cur['macd'] < cur['macd_signal'] and prev['macd'] >= prev['macd_signal']:
            mom_s -= 2
        if 30 < cur['rsi_14'] < 50 and cur['rsi_14'] > prev['rsi_14']:
            mom_s += 1
        elif 50 < cur['rsi_14'] < 70 and cur['rsi_14'] < prev['rsi_14']:
            mom_s -= 1
        if mom_s >= 2:
            bull.append(('Strong Momentum', 0.20))
        elif mom_s >= 1:
            bull.append(('Moderate Momentum', 0.10))
        elif mom_s <= -2:
            bear.append(('Weak Momentum', 0.20))

        # Mean reversion (15%)
        if cur['williams_r'] < -80:
            bull.append(('Oversold Bounce', 0.15))
        elif cur['williams_r'] > -20:
            bear.append(('Overbought Reversal', 0.15))

        # Volume (10%)
        if cur['vol_ratio'] > 1.5 and cur['close'] > prev['close']:
            bull.append(('Volume Confirmation', 0.10))
        elif cur['vol_ratio'] > 1.5 and cur['close'] < prev['close']:
            bear.append(('Volume Selling', 0.10))

        # Volatility/bands (10%)
        width = cur['bb_width'] if not pd.isna(cur['bb_width']) else 0.0
        denom = (cur['bb_upper_2'] - cur['bb_lower_2'])
        pos = (cur['close'] - cur['bb_lower_2']) / denom if denom and not pd.isna(denom) else 0.5
        if pos < 0.2 and width > 0.1:
            bull.append(('Volatility Squeeze', 0.10))
        elif pos > 0.8:
            bear.append(('Extended Move', 0.10))

        # Fundamentals (15%)
        f_score = self._fundamental_score(symbol)
        if f_score > 0.7:
            bull.append(('Strong Fundamentals', 0.15))
        elif f_score > 0.4:
            bull.append(('Good Fundamentals', 0.08))
        elif f_score < 0.3:
            bear.append(('Weak Fundamentals', 0.10))

        # Sentiment (5%)
        sent = self.data_provider.get_news_sentiment(symbol)
        if sent > 0.3:
            bull.append(('Positive Sentiment', 0.05))
        elif sent < -0.3:
            bear.append(('Negative Sentiment', 0.05))

        bull_score = sum(w for _, w in bull)
        bear_score = sum(w for _, w in bear)

        regime_adj = {
            'BULL': (1.1, 0.9),
            'BEAR': (0.9, 1.1),
            'VOLATILE': (0.8, 0.8),
            'NORMAL': (1.0, 1.0)
        }
        bmul, smul = regime_adj.get(self.market_regime, (1.0, 1.0))
        bull_score *= bmul
        bear_score *= smul

        action = 'hold'
        conf = max(bull_score, bear_score)
        reason = f'Bull {bull_score:.2f} vs Bear {bear_score:.2f}'
        if bull_score > bear_score and bull_score > self.min_confidence:
            action = 'buy'
        elif bear_score > bull_score and bear_score > self.min_confidence:
            action = 'sell'

        # ATR-based brackets for execution layer
        atr = cur.get('atr_14', np.nan)
        stop_price = None
        take_profit = None
        if not pd.isna(atr):
            if action == 'buy':
                stop_price = float(cur['close'] - 1.5 * atr)
                take_profit = float(cur['close'] + 2.5 * atr)
            elif action == 'sell':
                stop_price = float(cur['close'] + 1.5 * atr)
                take_profit = float(cur['close'] - 2.5 * atr)

        return Signal(action, float(np.clip(conf, 0.0, 0.99)), reason, self.market_regime, stop_price, take_profit)


class IntelligentRiskManager:
    """Kelly-lite, volatility targeting, and exposure caps."""

    def __init__(self) -> None:
        self.max_portfolio_risk = float(os.getenv('MAX_PORTFOLIO_RISK', '10')) / 100.0
        self.max_position_value = float(os.getenv('MAX_POSITION_SIZE', '5000'))
        self.max_daily_trades = int(os.getenv('MAX_DAILY_TRADES', '20'))
        self.daily_trades = 0
        self.last_reset = datetime.now().date()
        self.max_gross_exposure = float(os.getenv('MAX_GROSS_EXPOSURE', '1.0'))  # 1x NAV

    def reset_daily_counters(self) -> None:
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_trades = 0
            self.last_reset = today
            logger.info('Daily counters reset')

    def validate_trade(self, qty: int, price: float) -> bool:
        self.reset_daily_counters()
        if self.daily_trades >= self.max_daily_trades:
            return False
        est_cost = qty * price
        if est_cost > self.max_position_value:
            return False
        return True

    def kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        edge = win_rate * avg_win - (1 - win_rate) * avg_loss
        if avg_win <= 0:
            return 0.0
        k = edge / avg_win
        return float(np.clip(k, 0.0, 0.25))

    def position_size(self, signal_conf: float, account_value: float, volatility: float, price: float) -> int:
        # Kelly-lite
        k = self.kelly_fraction(win_rate=signal_conf, avg_win=0.02, avg_loss=0.01)
        # Vol targeting (reduce when vol high)
        vol_adj = max(0.1, 1.0 - min(volatility, 1.0))
        frac = k * vol_adj * (0.5 + 0.5 * signal_conf)
        risk_dollars = account_value * frac
        notional = min(risk_dollars, self.max_position_value)
        qty = max(0, int(notional // max(price, 1e-6)))
        return qty


class AdvancedTradingBot:
    """Execution layer with Alpaca and journaling."""

    def __init__(self) -> None:
        if tradeapi is None:
            raise RuntimeError('alpaca_trade_api not installed')
        self.api = tradeapi.REST(
            os.getenv('APCA_API_KEY_ID'),
            os.getenv('APCA_API_SECRET_KEY'),
            os.getenv('APCA_API_BASE_URL'),
            api_version='v2'
        )
        self.account = self.api.get_account()
        logger.info(f"Connected! Account: {self.account.id} BuyingPower: ${float(self.account.buying_power):,.2f}")
        self.strategy = AdvancedStrategy()
        self.risk = IntelligentRiskManager()
        self.spy_cache: Optional[pd.DataFrame] = None
        self.spy_last = None
        self.watchlist: Dict[str, List[str]] = {
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AMD', 'NFLX'],
            'financial': ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C'],
            'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
            'consumer': ['AMZN', 'WMT', 'HD', 'MCD', 'NKE'],
            'etfs': ['SPY', 'QQQ', 'IWM', 'VTI', 'XLF', 'XLK'],
            'energy': ['XOM', 'CVX', 'COP', 'EOG'],
            'industrial': ['BA', 'CAT', 'GE', 'MMM']
        }
        self.symbols = [s for lst in self.watchlist.values() for s in lst]
        self.journal_path = '/workspace/logs/trade_journal.jsonl'

    def _get_bars(self, symbol: str, timeframe: str = '5Min', limit: int = 1000) -> Optional[pd.DataFrame]:
        try:
            end = datetime.now()
            start = end - timedelta(days=30)
            bars = self.api.get_bars(symbol, timeframe, start=start.isoformat(), end=end.isoformat(), limit=limit).df
            return bars if len(bars) > 50 else None
        except Exception as exc:
            logger.error(f'Data fetch error {symbol}: {exc}')
            return None

    def _update_spy(self) -> None:
        try:
            if (self.spy_last is None) or ((datetime.now() - self.spy_last).seconds > 1800):
                self.spy_cache = self._get_bars('SPY', '15Min', 2000)
                self.spy_last = datetime.now()
                logger.info('SPY updated')
        except Exception as exc:
            logger.error(f'SPY update error: {exc}')

    def _journal(self, payload: Dict) -> None:
        try:
            with open(self.journal_path, 'a') as f:
                f.write(json.dumps(payload) + '\n')
        except Exception as exc:
            logger.warning(f'Journal write failed: {exc}')

    def _market_open(self) -> bool:
        try:
            return bool(self.api.get_clock().is_open)
        except Exception:
            return True

    def _symbol_price_vol(self, symbol: str) -> Tuple[float, float]:
        data = self._get_bars(symbol, '1Min', 120)
        if data is None or len(data) < 30:
            return (np.nan, 0.3)
        price = float(data['close'].iloc[-1])
        vol = float(data['close'].pct_change().std() * math.sqrt(252))
        return (price, vol)

    def _submit_bracket(self, symbol: str, side: str, qty: int, stop: Optional[float], take_profit: Optional[float]) -> Optional[Dict]:
        try:
            if qty <= 0:
                return None
            # Prefer bracket if prices exist, fallback to market order
            if stop and take_profit:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=str(qty),
                    side=side,
                    type='market',
                    time_in_force='day',
                    order_class='bracket',
                    stop_loss={'stop_price': f'{stop:.2f}'},
                    take_profit={'limit_price': f'{take_profit:.2f}'}
                )
            else:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=str(qty),
                    side=side,
                    type='market',
                    time_in_force='day'
                )
            self.risk.daily_trades += 1
            return {'id': getattr(order, 'id', None), 'qty': qty}
        except Exception as exc:
            logger.error(f'Order failed {symbol}: {exc}')
            return None

    def run_cycle(self) -> None:
        if not self._market_open():
            logger.info('Market closed - idle')
            return
        self._update_spy()
        acct = self.api.get_account()
        buying_power = float(acct.buying_power)
        portfolio_value = float(acct.portfolio_value)
        logger.info(f'Portfolio ${portfolio_value:,.2f} | BP ${buying_power:,.2f}')
        self.risk.reset_daily_counters()

        analyzed = 0
        executed = 0
        for symbol in self.symbols:
            if self.risk.daily_trades >= self.risk.max_daily_trades:
                logger.info('Daily trade limit reached')
                break
            df = self._get_bars(symbol, '5Min', 1200)
            if df is None or len(df) < 200:
                continue
            sig = self.strategy.generate_signal(df, symbol, self.spy_cache)
            analyzed += 1
            logger.info(f"{symbol}: {sig.action.upper()} conf={sig.confidence:.2f} {sig.reason} regime={sig.market_regime}")
            if sig.action in ('buy', 'sell') and sig.confidence >= 0.70:
                price, vol = self._symbol_price_vol(symbol)
                if np.isnan(price):
                    continue
                qty = self.risk.position_size(sig.confidence, buying_power, vol, price)
                if not self.risk.validate_trade(qty, price):
                    continue
                side = 'buy' if sig.action == 'buy' else 'sell'
                res = self._submit_bracket(symbol, side, qty, sig.stop_price, sig.take_profit_price)
                if res:
                    executed += 1
                    self._journal({
                        'ts': datetime.utcnow().isoformat(),
                        'symbol': symbol,
                        'side': side,
                        'qty': qty,
                        'price_ref': price,
                        'confidence': sig.confidence,
                        'regime': sig.market_regime,
                        'stop': sig.stop_price,
                        'take_profit': sig.take_profit_price,
                        'reason': sig.reason
                    })
                time.sleep(1.5)
        logger.info(f'Cycle: analyzed={analyzed} executed={executed} regime={self.strategy.market_regime}')

    def run(self) -> None:
        logger.info('ADVANCED TRADING BOT STARTED!')
        while True:
            try:
                self.run_cycle()
                sleep_s = 30 if self.strategy.market_regime == 'VOLATILE' else 45 if self.strategy.market_regime == 'BULL' else 60
                logger.info(f'Next cycle in {sleep_s}s')
                time.sleep(sleep_s)
            except KeyboardInterrupt:
                logger.info('Stopped by user')
                break
            except Exception as exc:
                logger.error(f'Critical error: {exc}')
                logger.info('Recovering in 60s...')
                time.sleep(60)


def main() -> None:
    required = ['APCA_API_KEY_ID', 'APCA_API_SECRET_KEY', 'APCA_API_BASE_URL']
    missing = [k for k in required if not os.getenv(k) or 'your_' in os.getenv(k, '')]
    if missing:
        raise RuntimeError(f'Missing required env: {missing}')
    optional = ['ALPHA_VANTAGE_API_KEY', 'NEWS_API_KEY']
    for k in optional:
        if not os.getenv(k):
            logger.warning(f'{k} not set - reduced features')
    bot = AdvancedTradingBot()
    bot.run()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Critical Error: {exc}')
        print('\nSetup Instructions:')
        print('1. Get Alpaca API keys from https://alpaca.markets')
        print('2. Get Alpha Vantage API key from https://www.alphavantage.co/support/#api-key')
        print('3. Get News API key from https://newsapi.org/')
        print('4. Edit /workspace/config/.env with your API keys')
        print('5. Run: python /workspace/bot/phd_trading_bot.py')