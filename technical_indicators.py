"""
Technical indicators module for enhanced binary options bot
"""
import numpy as np
from collections import deque
import logging
import numpy as np

class TechnicalIndicators:
    def __init__(self, config):
        """Initialize technical indicators with configuration"""
        self.config = config
        
        # Store price history
        self.price_history = deque(maxlen=100)
        self.tick_history = deque(maxlen=100)
        
        # Initialize indicators storage with default values
        self.indicators = {
            'rsi': 50.0,  # Neutral RSI
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'macd_histogram_prev': 0.0,
            'bollinger_upper': 0.0,
            'bollinger_middle': 0.0,
            'bollinger_lower': 0.0,
            'trend': 'neutral',
            'volatility': 'medium',
            'pattern': 'none',
            'rsi_history': {}
        }
        
        # Pre-initialize RSI history with neutral values
        for i in range(-10, 0):
            self.indicators['rsi_history'][i] = 50.0
        
        self.logger = logging.getLogger('technical_indicators')
    
    def add_price(self, price, tick_data=None):
        """
        Add a new price point to history and update indicators
        
        Args:
            price (float): Current price
            tick_data (dict): Full tick data if available
        """
        self.price_history.append(price)
        if tick_data:
            self.tick_history.append(tick_data)
            
        # Update indicators if we have enough data
        if len(self.price_history) >= self.config['TECHNICAL_ANALYSIS']['min_data_points']:
            self._calculate_indicators()
            return True
        return False
    
    def _calculate_indicators(self):
        """Calculate all technical indicators based on current price history"""
        self._calculate_rsi()
        self._calculate_macd()
        self._calculate_bollinger_bands()
        self._analyze_trend()
        self._calculate_volatility()
        self._detect_patterns()
        
    def _calculate_rsi(self):
        """Calculate Relative Strength Index"""
        if len(self.price_history) < self.config['TECHNICAL_ANALYSIS']['rsi_period'] + 1:
            self.indicators['rsi'] = 50  # Neutral value when not enough data
            return
            
        # Convert deque to numpy array for calculations
        prices = np.array(list(self.price_history))
        deltas = np.diff(prices)
        
        # Calculate gains and losses
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        period = self.config['TECHNICAL_ANALYSIS']['rsi_period']
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if len(deltas) > period:
            # Calculate smoothed averages
            for i in range(period, len(deltas)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
                
        # Calculate RSI
        if avg_loss == 0:
            self.indicators['rsi'] = 100
        else:
            rs = avg_gain / avg_loss
            self.indicators['rsi'] = 100 - (100 / (1 + rs))
    
    def _calculate_macd(self):
        """Calculate Moving Average Convergence Divergence"""
        if len(self.price_history) < self.config['TECHNICAL_ANALYSIS']['macd_slow'] + 1:
            self.indicators['macd'] = 0
            self.indicators['macd_signal'] = 0
            self.indicators['macd_histogram'] = 0
            return
            
        prices = np.array(list(self.price_history))
        
        # Calculate EMAs
        fast_period = self.config['TECHNICAL_ANALYSIS']['macd_fast']
        slow_period = self.config['TECHNICAL_ANALYSIS']['macd_slow']
        signal_period = self.config['TECHNICAL_ANALYSIS']['macd_signal']
        
        # Simple implementation of EMA
        fast_ema = self._calculate_ema(prices, fast_period)
        slow_ema = self._calculate_ema(prices, slow_period)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD line)
        signal_line = self._calculate_ema(macd_line, signal_period)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        self.indicators['macd'] = macd_line[-1] if len(macd_line) > 0 else 0
        self.indicators['macd_signal'] = signal_line[-1] if len(signal_line) > 0 else 0
        self.indicators['macd_histogram'] = histogram[-1] if len(histogram) > 0 else 0
    
    def _calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return np.array([0])
            
        ema = np.zeros_like(data)
        # Start with SMA
        ema[:period] = np.mean(data[:period])
        
        # Calculate multiplier
        multiplier = 2 / (period + 1)
        
        # Calculate EMA
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
            
        return ema
    
    def _calculate_bollinger_bands(self):
        """Calculate Bollinger Bands"""
        if len(self.price_history) < self.config['TECHNICAL_ANALYSIS']['bollinger_period']:
            # Set default values
            self.indicators['bollinger_upper'] = max(self.price_history) if self.price_history else 0
            self.indicators['bollinger_middle'] = np.mean(list(self.price_history)) if self.price_history else 0
            self.indicators['bollinger_lower'] = min(self.price_history) if self.price_history else 0
            return
            
        prices = np.array(list(self.price_history))
        period = self.config['TECHNICAL_ANALYSIS']['bollinger_period']
        std_dev = self.config['TECHNICAL_ANALYSIS']['bollinger_std']
        
        # Calculate middle band (SMA)
        middle_band = np.mean(prices[-period:])
        
        # Calculate standard deviation
        sigma = np.std(prices[-period:])
        
        # Calculate upper and lower bands
        upper_band = middle_band + (sigma * std_dev)
        lower_band = middle_band - (sigma * std_dev)
        
        self.indicators['bollinger_upper'] = upper_band
        self.indicators['bollinger_middle'] = middle_band
        self.indicators['bollinger_lower'] = lower_band
    
    def _analyze_trend(self):
        """Analyze current price trend"""
        if len(self.price_history) < 10:
            self.indicators['trend'] = 'neutral'
            return
            
        prices = list(self.price_history)
        
        # Simple trend detection using linear regression slope
        x = np.arange(len(prices))
        y = np.array(prices)
        
        # Calculate slope using linear regression
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
        
        # Determine trend based on slope
        threshold = 0.01  # Threshold for significance
        if slope > threshold:
            self.indicators['trend'] = 'uptrend'
        elif slope < -threshold:
            self.indicators['trend'] = 'downtrend'
        else:
            self.indicators['trend'] = 'neutral'
    
    def _calculate_volatility(self):
        """Calculate price volatility"""
        if len(self.price_history) < 10:
            self.indicators['volatility'] = 'medium'
            return
            
        prices = np.array(list(self.price_history))
        
        # Calculate percentage changes
        pct_changes = np.diff(prices) / prices[:-1] * 100
        
        # Calculate standard deviation of percentage changes
        volatility = np.std(pct_changes)
        
        # Categorize volatility
        if volatility < 0.1:
            self.indicators['volatility'] = 'low'
        elif volatility > 0.3:
            self.indicators['volatility'] = 'high'
        else:
            self.indicators['volatility'] = 'medium'
    
    def _detect_patterns(self):
        """Detect price patterns"""
        if len(self.price_history) < 20:
            self.indicators['pattern'] = 'none'
            return
            
        prices = list(self.price_history)
        
        # Simple pattern detection
        latest_prices = prices[-5:]
        
        # Detect patterns based on recent price movements
        consecutive_up = all(latest_prices[i] > latest_prices[i-1] for i in range(1, len(latest_prices)))
        consecutive_down = all(latest_prices[i] < latest_prices[i-1] for i in range(1, len(latest_prices)))
        
        if consecutive_up:
            self.indicators['pattern'] = 'consecutive_up'
        elif consecutive_down:
            self.indicators['pattern'] = 'consecutive_down'
        else:
            # Check for potential reversal
            if prices[-1] > prices[-2] > prices[-3] and prices[-4] < prices[-3] and prices[-5] < prices[-4]:
                self.indicators['pattern'] = 'potential_reversal_up'
            elif prices[-1] < prices[-2] < prices[-3] and prices[-4] > prices[-3] and prices[-5] > prices[-4]:
                self.indicators['pattern'] = 'potential_reversal_down'
            else:
                self.indicators['pattern'] = 'none'
    
    def get_trade_signal(self):
        """
        Generate a trade signal based on enhanced technical indicators
        with optimized entry points for improved win rate
        
        Returns:
            tuple: (signal, confidence, reason)
                signal: 'CALL', 'PUT', or 'wait'
                confidence: 0.0-1.0 indicating confidence level
                reason: Explanation of the signal
        """
        if not self.indicators['rsi']:
            return ('wait', 0, "Insufficient data for analysis")
            
        # Get indicator values
        rsi = self.indicators['rsi']
        macd = self.indicators['macd']
        macd_signal = self.indicators['macd_signal']
        macd_histogram = self.indicators['macd_histogram']
        trend = self.indicators['trend']
        pattern = self.indicators['pattern']
        bollinger_upper = self.indicators['bollinger_upper']
        bollinger_middle = self.indicators['bollinger_middle']
        bollinger_lower = self.indicators['bollinger_lower']
        volatility = self.indicators['volatility']
        
        # Get the most recent prices
        prices = list(self.price_history)
        current_price = prices[-1] if prices else None
        
        if not current_price:
            return ('wait', 0, "No current price data")
        
        # Track signal confidence and reasons
        signal = 'wait'
        confidence = 0.5
        reasons = []
        
        # ---- ENHANCED STRATEGY 1: MEAN REVERSION STRATEGY ----
        # For highly volatile markets, look for extreme price movements
        # and bet on reversion to the mean
        
        if volatility == 'high':
            # If price is near the upper bollinger band + overbought RSI, PUT signal
            if current_price > bollinger_upper * 0.98 and rsi > 70:
                signal = 'PUT'
                confidence += 0.2
                reasons.append(f"Mean reversion: Price near upper band ({current_price:.4f}/{bollinger_upper:.4f}) with overbought RSI ({rsi:.2f})")
            
            # If price is near the lower bollinger band + oversold RSI, CALL signal
            elif current_price < bollinger_lower * 1.02 and rsi < 30:
                signal = 'CALL'
                confidence += 0.2
                reasons.append(f"Mean reversion: Price near lower band ({current_price:.4f}/{bollinger_lower:.4f}) with oversold RSI ({rsi:.2f})")
        
        # ---- ENHANCED STRATEGY 2: MOMENTUM STRATEGY ----
        # Look for strong directional movements with confirming indicators
        
        # Check for MACD, RSI and trend alignment for strong signals
        macd_increasing = macd_histogram > self.indicators.get('macd_histogram_prev', 0)
        macd_decreasing = macd_histogram < self.indicators.get('macd_histogram_prev', 0)
        
        # Strong bullish momentum
        if (macd > macd_signal and macd_increasing and trend == 'uptrend' and 
            (40 < rsi < 80) and pattern in ['consecutive_up', 'potential_reversal_up']):
            if signal == 'wait' or signal == 'CALL':
                signal = 'CALL'
                confidence += 0.25
                reasons.append("Strong bullish momentum with trend confirmation")
            
        # Strong bearish momentum
        elif (macd < macd_signal and macd_decreasing and trend == 'downtrend' and 
              (20 < rsi < 60) and pattern in ['consecutive_down', 'potential_reversal_down']):
            if signal == 'wait' or signal == 'PUT':
                signal = 'PUT'
                confidence += 0.25
                reasons.append("Strong bearish momentum with trend confirmation")
        
        # ---- ENHANCED STRATEGY 3: BREAKOUT DETECTION ----
        # Identify breakout movements for short-term trades
        
        # Check for price breaking out of recent range
        price_std = np.std(prices[-10:]) if len(prices) >= 10 else 0
        price_mean = np.mean(prices[-10:]) if len(prices) >= 10 else current_price
        
        # Define breakout thresholds (2 standard deviations)
        breakout_threshold = price_std * 1.5
        
        # Bullish breakout detection
        if (current_price > price_mean + breakout_threshold and
            macd_increasing and rsi > 50 and rsi < 80):
            if signal == 'wait' or confidence < 0.7:  # Breakout signal is strong
                signal = 'CALL'
                confidence = max(confidence, 0.7)
                reasons.append(f"Bullish breakout: Price exceeded upper threshold ({current_price:.4f} > {price_mean + breakout_threshold:.4f})")
        
        # Bearish breakout detection
        elif (current_price < price_mean - breakout_threshold and
              macd_decreasing and rsi < 50 and rsi > 20):
            if signal == 'wait' or confidence < 0.7:  # Breakout signal is strong
                signal = 'PUT'
                confidence = max(confidence, 0.7)
                reasons.append(f"Bearish breakout: Price below lower threshold ({current_price:.4f} < {price_mean - breakout_threshold:.4f})")
        
        # ---- ENHANCED STRATEGY 4: DIVERGENCE DETECTION ----
        # Look for divergences between price and indicators
        
        # Simple divergence check (can be improved with more sophisticated detection)
        if len(prices) >= 5:
            price_direction = prices[-1] > prices[-5]  # True if price is up over last 5 ticks
            
            # Bearish divergence (price up but RSI down)
            rsi_values = [self.indicators.get('rsi_history', {}).get(i, 50) for i in range(-5, 0)]
            if price_direction and rsi_values and rsi_values[0] > rsi_values[-1] and rsi > 60:
                signal = 'PUT'
                confidence += 0.15
                reasons.append("Bearish divergence: Price up but RSI down")
            
            # Bullish divergence (price down but RSI up)
            elif not price_direction and rsi_values and rsi_values[0] < rsi_values[-1] and rsi < 40:
                signal = 'CALL'
                confidence += 0.15
                reasons.append("Bullish divergence: Price down but RSI up")
        
        # ---- FALLBACK: BASIC INDICATOR ANALYSIS ----
        # Use basic indicator analysis if no strong signals from enhanced strategies
        
        if signal == 'wait':
            # RSI analysis - more sensitive thresholds and higher confidence
            if rsi <= 28:  # More extreme oversold (was 30)
                signal = 'CALL'
                confidence += 0.15  # Increased from 0.1
                reasons.append(f"Strong oversold RSI ({rsi:.2f})")
            elif rsi >= 72:  # More extreme overbought (was 70)
                signal = 'PUT'
                confidence += 0.15  # Increased from 0.1
                reasons.append(f"Strong overbought RSI ({rsi:.2f})")
            
            # MACD analysis - added signal strength measurement
            macd_strength = abs(macd - macd_signal) / max(0.0001, abs(macd_signal))  # Relative strength
            
            if macd > macd_signal and macd_histogram > 0 and macd_increasing:
                if signal == 'wait' or signal == 'CALL':
                    signal = 'CALL'
                    confidence += min(0.2, 0.15 + macd_strength * 0.1)  # Adjust by signal strength
                    reasons.append(f"MACD bullish signal (strength: {macd_strength:.2f})")
                    
            elif macd < macd_signal and macd_histogram < 0 and macd_decreasing:
                if signal == 'wait' or signal == 'PUT':
                    signal = 'PUT'
                    confidence += min(0.2, 0.15 + macd_strength * 0.1)  # Adjust by signal strength
                    reasons.append(f"MACD bearish signal (strength: {macd_strength:.2f})")
        
        # ---- FINAL CONFIRMATIONS AND ADJUSTMENTS ----
        
        # Verify trend alignment for higher confidence
        if signal == 'CALL' and trend == 'uptrend':
            confidence += 0.1
            reasons.append("Uptrend confirmation")
        elif signal == 'PUT' and trend == 'downtrend':
            confidence += 0.1
            reasons.append("Downtrend confirmation")
        
        # Pattern confirmation with higher confidence boost
        if signal == 'CALL' and pattern == 'consecutive_up':
            confidence += 0.1  # Increased from 0.05
            reasons.append("Consecutive up pattern confirmation")
        elif signal == 'PUT' and pattern == 'consecutive_down':
            confidence += 0.1  # Increased from 0.05
            reasons.append("Consecutive down pattern confirmation")
        elif signal == 'CALL' and pattern == 'potential_reversal_up':
            confidence += 0.15  # Increased from 0.1
            reasons.append("Potential reversal up pattern")
        elif signal == 'PUT' and pattern == 'potential_reversal_down':
            confidence += 0.15  # Increased from 0.1
            reasons.append("Potential reversal down pattern")
        
        # If signals are conflicting, reduce confidence or wait
        if (signal == 'CALL' and trend == 'downtrend') or (signal == 'PUT' and trend == 'uptrend'):
            if confidence < 0.7:  # Only override strong signals
                confidence -= 0.15  # Increase penalty
                reasons.append(f"Warning: Signal conflicts with {trend}")
                
                # If confidence becomes too low, wait
                if confidence < 0.55:
                    signal = 'wait'
                    reasons = ["Conflicting signals with low confidence"]
        
        # Store RSI history for divergence detection
        if 'rsi_history' not in self.indicators:
            self.indicators['rsi_history'] = {}
        
        # Shift history
        for i in range(-10, -1):
            self.indicators['rsi_history'][i] = self.indicators['rsi_history'].get(i+1, 50)
        self.indicators['rsi_history'][-1] = rsi
            
        # Store current values for next comparison
        self.indicators['macd_histogram_prev'] = macd_histogram
        
        # Ensure confidence is within range and adjust final threshold for trade
        confidence = max(0.0, min(1.0, confidence))
        
        # Higher confidence threshold for actual trades (0.6 -> 0.65)
        if confidence < 0.65 and signal != 'wait':
            signal = 'wait'
            reasons = ["Signal confidence below threshold"]
            confidence = 0.0
        
        # Return the signal with confidence and concatenated reasons
        return (signal, confidence, "; ".join(reasons))
    
    def get_indicators_summary(self):
        """Get a summary of current indicator values"""
        return {
            'rsi': round(self.indicators['rsi'], 2) if self.indicators['rsi'] is not None else None,
            'macd': round(self.indicators['macd'], 4) if self.indicators['macd'] is not None else None,
            'macd_signal': round(self.indicators['macd_signal'], 4) if self.indicators['macd_signal'] is not None else None,
            'trend': self.indicators['trend'],
            'volatility': self.indicators['volatility'],
            'pattern': self.indicators['pattern'],
            'price': self.price_history[-1] if self.price_history else None,
        }
