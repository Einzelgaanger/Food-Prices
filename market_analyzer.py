"""
Market analyzer module for enhanced binary options bot
"""
from collections import defaultdict
import random
import logging
from datetime import datetime, timedelta
import numpy as np

class MarketAnalyzer:
    def __init__(self, config):
        """Initialize market analyzer with configuration"""
        self.config = config
        self.market_data = defaultdict(dict)
        self.best_markets = []
        self.market_rankings = {}
        
        # Market analysis metrics
        self.volatility_rankings = {}
        self.win_rate_by_market = defaultdict(lambda: {"wins": 0, "losses": 0})
        self.time_analysis = defaultdict(lambda: {"wins": 0, "losses": 0})
        
        # Initialize all available markets
        self.all_markets = []
        for category in config['MARKETS']:
            self.all_markets.extend(config['MARKETS'][category])
        
        self.logger = logging.getLogger('market_analyzer')
    
    def update_market_data(self, market, tick_data):
        """
        Update market data with new tick
        
        Args:
            market (str): Market symbol
            tick_data (dict): Tick data
        """
        if 'price' not in tick_data:
            return
            
        # Initialize market entry if it doesn't exist
        if 'ticks' not in self.market_data[market]:
            self.market_data[market]['ticks'] = []
            self.market_data[market]['volatility'] = 0
            self.market_data[market]['last_update'] = datetime.now()
        
        # Add tick data
        self.market_data[market]['ticks'].append(tick_data)
        
        # Keep only the last 100 ticks to avoid memory issues
        if len(self.market_data[market]['ticks']) > 100:
            self.market_data[market]['ticks'] = self.market_data[market]['ticks'][-100:]
        
        # Update volatility calculation if we have enough data
        if len(self.market_data[market]['ticks']) >= 10:
            self._update_volatility(market)
        
        # Update last update time
        self.market_data[market]['last_update'] = datetime.now()
    
    def _update_volatility(self, market):
        """
        Calculate market volatility based on recent ticks
        
        Args:
            market (str): Market symbol
        """
        ticks = self.market_data[market]['ticks']
        prices = [tick['price'] for tick in ticks]
        
        # Calculate percentage changes
        pct_changes = np.diff(prices) / prices[:-1] * 100
        
        # Calculate volatility as standard deviation of percentage changes
        volatility = np.std(pct_changes)
        
        self.market_data[market]['volatility'] = volatility
        self.logger.debug(f"Updated volatility for {market}: {volatility:.4f}%")
    
    def update_win_loss(self, market, won):
        """
        Update win/loss statistics for a market
        
        Args:
            market (str): Market symbol
            won (bool): Whether the trade was won
        """
        if won:
            self.win_rate_by_market[market]["wins"] += 1
        else:
            self.win_rate_by_market[market]["losses"] += 1
        
        # Also record by time of day (hour)
        hour = datetime.now().hour
        if won:
            self.time_analysis[hour]["wins"] += 1
        else:
            self.time_analysis[hour]["losses"] += 1
    
    def get_win_rate(self, market):
        """
        Get win rate for a specific market
        
        Args:
            market (str): Market symbol
            
        Returns:
            float: Win rate (0.0-1.0)
        """
        stats = self.win_rate_by_market[market]
        total = stats["wins"] + stats["losses"]
        
        if total == 0:
            return 0.5  # Default to neutral when no data
        
        return stats["wins"] / total
    
    def get_current_hour_win_rate(self):
        """
        Get win rate for the current hour
        
        Returns:
            float: Win rate (0.0-1.0)
        """
        hour = datetime.now().hour
        stats = self.time_analysis[hour]
        total = stats["wins"] + stats["losses"]
        
        if total == 0:
            return 0.5  # Default to neutral when no data
        
        return stats["wins"] / total
    
    def rank_markets(self):
        """
        Enhanced market ranking system that identifies optimal trading conditions
        using multiple factors including volatility, win rate, time patterns,
        price action, and market-specific characteristics
        
        Returns:
            list: Ranked list of market symbols optimized for trading
        """
        if not self.market_data:
            self.logger.warning("No market data available for ranking")
            return self.all_markets.copy()  # Return all markets if no data
        
        # Initialize score components
        scores = {}
        volatility_scores = {}
        win_rate_scores = {}
        pattern_scores = {}
        time_scores = {}
        price_action_scores = {}
        
        # Current hour for time-based analysis
        current_hour = datetime.now().hour
        
        # Calculate scores for each market using multiple metrics
        for market in self.market_data:
            # Skip markets with insufficient data
            if 'volatility' not in self.market_data[market] or not self.market_data[market]['ticks']:
                continue
                
            # --- Basic metrics ---
            volatility = self.market_data[market]['volatility']
            win_rate = self.get_win_rate(market)
            
            # Calculate recency (how recently we updated this market)
            last_update = self.market_data[market]['last_update']
            recency = max(0, 1 - (datetime.now() - last_update).total_seconds() / 3600)
            
            # --- Enhanced volatility scoring ---
            # Instead of linear scoring, use a bell curve that favors medium-high volatility
            # Too low volatility = hard to predict, too high = erratic
            optimal_volatility = 0.3  # Optimal volatility percentage for trading
            volatility_distance = abs(volatility - optimal_volatility) / optimal_volatility
            volatility_score = max(0, 1 - volatility_distance)
            
            # --- Win rate scoring with confidence adjustment ---
            # Favor markets with consistent win rates and sufficient data
            market_trades = self.win_rate_by_market[market]["wins"] + self.win_rate_by_market[market]["losses"]
            win_rate_confidence = min(1.0, market_trades / 20)  # Scale up to 20 trades for max confidence
            
            # Scale win rate to emphasize differences (0.5=neutral, 0.6+=good, 0.7+=excellent)
            if win_rate > 0.5:
                # Exponential scaling for win rates above 0.5 to reward higher win rates more
                win_rate_score = 0.5 + 0.5 * ((win_rate - 0.5) / 0.5) ** 1.5 
            else:
                # Linear scaling for win rates below 0.5
                win_rate_score = win_rate
                
            # Apply confidence factor
            win_rate_score = 0.5 + (win_rate_score - 0.5) * win_rate_confidence
            
            # --- Time-based pattern scoring ---
            # Check if this market performs well during the current hour
            hour_stats = {hr: self.time_analysis.get(hr, {"wins": 0, "losses": 0}) for hr in range(24)}
            
            # Calculate win rates by hour with sufficient data
            hour_win_rates = {}
            for hr, stats in hour_stats.items():
                total = stats["wins"] + stats["losses"]
                if total >= 5:  # Minimum trades to consider
                    hour_win_rates[hr] = stats["wins"] / total
                else:
                    hour_win_rates[hr] = 0.5  # Neutral when insufficient data
            
            # Score current hour performance
            current_hour_win_rate = hour_win_rates.get(current_hour, 0.5)
            time_score = max(0.4, min(1.0, current_hour_win_rate / 0.6))  # Scale: 0.4-1.0
            
            # --- Price action scoring ---
            # Analyze recent price movements for patterns
            ticks = self.market_data[market]['ticks']
            if len(ticks) >= 20:
                prices = [tick.get('price', tick.get('quote', 0)) for tick in ticks[-20:]]
                
                # Calculate price trend strength
                trend_strength = 0.5  # Default neutral value
                pattern_value = 0.5  # Default neutral value for pattern detection
                
                if len(prices) >= 5:
                    # Simple linear regression for trend detection
                    x = np.array(range(len(prices)))
                    y = np.array(prices)
                    slope = np.polyfit(x, y, 1)[0]
                    
                    # Normalize trend strength
                    avg_price = np.mean(prices)
                    normalized_slope = abs(slope / avg_price * 100)
                    
                    # Strong trends (either direction) are good for trading
                    trend_strength = min(1.0, normalized_slope / 0.2)  # Cap at 1.0
                
                # Simple pattern detection - looking for consolidation followed by movement
                if len(prices) >= 10:
                    recent_volatility = np.std(prices[-5:]) / np.mean(prices[-5:])
                    past_volatility = np.std(prices[-10:-5]) / np.mean(prices[-10:-5])
                    
                    # Volatility expansion (breakout) or contraction (consolidation)
                    vol_ratio = recent_volatility / max(0.0001, past_volatility)
                    
                    if vol_ratio > 1.5:  # Volatility expansion - good for trading
                        pattern_value = 0.8
                    elif vol_ratio < 0.7:  # Volatility contraction - potential setup
                        pattern_value = 0.7
                
                price_action_score = (trend_strength * 0.6) + (pattern_value * 0.4)
            else:
                price_action_score = 0.5  # Neutral when insufficient data
            
            # --- Market-specific characteristics ---
            # Some markets may have inherent advantages for binary options
            market_boost = 0
            
            # Favor high-volatility indices for binary options
            if "V" in market:  # Volatility index
                market_boost += 0.1
                
            # Crash/Boom indices have clear patterns
            if "CRASH" in market or "BOOM" in market:
                market_boost += 0.05
            
            # --- Combine all scoring factors ---
            volatility_scores[market] = volatility_score
            win_rate_scores[market] = win_rate_score
            time_scores[market] = time_score
            price_action_scores[market] = price_action_score
            pattern_scores[market] = pattern_value
            
            # Weighted final score with optimized weights for binary options trading
            score = (
                (0.30 * win_rate_score) +      # Win rate is critical
                (0.25 * volatility_score) +    # Volatility for good entry points
                (0.20 * price_action_score) +  # Price action for timing
                (0.15 * time_score) +          # Time patterns
                (0.10 * recency) +             # Data recency
                market_boost                   # Market-specific advantages
            )
            
            scores[market] = score
            
            # Log detailed market analysis periodically
            if random.random() < 0.1:  # 10% chance to log detailed analysis to avoid excessive logging
                self.logger.info(f"Market {market} analysis: "
                                f"Win rate: {win_rate:.2f} (score: {win_rate_score:.2f}), "
                                f"Volatility: {volatility:.4f} (score: {volatility_score:.2f}), "
                                f"Current hour: {current_hour} (score: {time_score:.2f}), "
                                f"Price action: {price_action_score:.2f}, "
                                f"Final score: {score:.2f}")
        
        # Sort markets by score (descending)
        ranked_markets = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Store detailed rankings for reference
        self.market_rankings = {
            market: {
                'rank': i+1, 
                'total_score': scores[market],
                'win_rate': win_rate_scores.get(market, 0),
                'volatility': volatility_scores.get(market, 0),
                'time_pattern': time_scores.get(market, 0),
                'price_action': price_action_scores.get(market, 0)
            } for i, market in enumerate(ranked_markets)
        }
        
        # If we don't have enough ranked markets, add some random ones from all markets
        if len(ranked_markets) < 5:
            remaining_markets = [m for m in self.all_markets if m not in ranked_markets]
            random.shuffle(remaining_markets)
            ranked_markets.extend(remaining_markets)
        
        # Store top markets but give slight preference to markets with recent winning trades
        candidate_markets = ranked_markets[:8]  # Consider top 8 as candidates
        
        # Identify markets with recent wins within candidate markets
        recent_winners = []
        for market in candidate_markets:
            market_stats = self.win_rate_by_market[market]
            if market_stats["wins"] > market_stats["losses"] * 1.2:  # 20% better win rate
                recent_winners.append(market)
        
        # Create final list with preference for recent winners, but maintaining overall ranking
        if recent_winners:
            # Move some recent winners to the top positions if they're not already there
            for winner in recent_winners[:2]:  # Consider up to 2 recent winners
                if winner in candidate_markets[3:]:  # If not already in top 3
                    candidate_markets.remove(winner)
                    candidate_markets.insert(random.randint(0, 2), winner)  # Insert in top 3
        
        self.best_markets = candidate_markets[:5]  # Store top 5 markets
        
        # Log top markets
        self.logger.info(f"Market ranking updated. Top markets: {', '.join(self.best_markets)}")
        
        return ranked_markets
    
    def get_best_market(self):
        """
        Get the best market to trade on
        
        Returns:
            str: Market symbol
        """
        if not self.best_markets:
            self.rank_markets()
            
        if not self.best_markets:
            # If still no markets, return a random one from all markets
            return random.choice(self.all_markets)
            
        # Return the top market with some randomness (20% chance to pick a different one from top 5)
        if random.random() < 0.8:
            return self.best_markets[0]
        else:
            return random.choice(self.best_markets[1:5] if len(self.best_markets) >= 5 else self.best_markets)
    
    def get_ideal_contract_settings(self, market):
        """
        Get ideal contract settings for a market based on analysis
        
        Args:
            market (str): Market symbol
            
        Returns:
            dict: Contract settings
        """
        # Check if we have volatility data
        if market in self.market_data and 'volatility' in self.market_data[market]:
            volatility = self.market_data[market]['volatility']
        else:
            volatility = 0.2  # Default medium volatility
            
        # Base settings
        settings = {
            "duration": 1,
            "duration_unit": "t",  # t=ticks, s=seconds, m=minutes
        }
        
        # Adjust duration based on volatility
        if volatility < 0.1:  # Low volatility
            # For low volatility, longer durations may be better
            if 'price' in market:  # Volatility indices like R_100
                settings["duration"] = 2
                settings["duration_unit"] = "t"
            else:  # Other indices
                settings["duration"] = 15
                settings["duration_unit"] = "s"
        elif volatility > 0.3:  # High volatility
            # For high volatility, shorter durations to capture quick movements
            settings["duration"] = 1
            settings["duration_unit"] = "t"
        else:  # Medium volatility
            # Balanced approach
            settings["duration"] = 1  
            settings["duration_unit"] = "t"
                
        return settings
    
    def get_current_market_conditions(self):
        """
        Get a summary of current market conditions
        
        Returns:
            dict: Market conditions
        """
        # Analyze best times to trade based on win rate by hour
        best_hours = []
        for hour, stats in self.time_analysis.items():
            total = stats["wins"] + stats["losses"]
            if total >= 5:  # Only consider hours with sufficient data
                win_rate = stats["wins"] / total
                if win_rate > 0.55:  # Only hours with above average win rate
                    best_hours.append((hour, win_rate))
        
        # Sort by win rate
        best_hours.sort(key=lambda x: x[1], reverse=True)
        
        # Get current hour win rate
        current_hour_win_rate = self.get_current_hour_win_rate()
        
        return {
            "best_markets": self.best_markets if self.best_markets else ["No data yet"],
            "best_hours": [f"{hour}:00 ({win_rate:.2f})" for hour, win_rate in best_hours[:3]],
            "current_hour_win_rate": current_hour_win_rate,
            "is_favorable_time": current_hour_win_rate > 0.55
        }
