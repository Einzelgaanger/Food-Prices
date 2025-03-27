"""
Risk management module for enhanced binary options bot
"""
import logging
from datetime import datetime, timedelta
import numpy as np

class RiskManager:
    def __init__(self, config):
        """Initialize risk manager with configuration"""
        self.config = config
        self.base_stake = config['RISK_MANAGEMENT']['base_stake']
        self.max_stake = config['RISK_MANAGEMENT']['max_stake']
        self.max_daily_loss = config['TRADING']['max_daily_loss']
        self.profit_target = config['TRADING']['profit_target']
        
        # Trading session stats
        self.session_start = datetime.now()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        self.daily_profit = 0
        self.total_invested = 0
        self.total_payout = 0
        self.win_rate = 0.5  # Start with neutral win rate assumption
        
        # Trade history
        self.trade_history = []
        
        # Stake progression
        self.current_stake = self.base_stake
        self.stake_history = []
        
        # Session tracking
        self.daily_loss_limit_reached = False
        self.profit_target_reached = False
        self.recovery_mode = False
        
        # Market condition attributes (set by the main bot)
        self.market_volatility = 'medium'  # Options: 'low', 'medium', 'high'
        self.favorable_time = False        # Whether current hour has good win rate historically
        self.current_hour_win_rate = 0.5   # Current hour's historical win rate
        
        self.logger = logging.getLogger('risk_manager')
    
    def calculate_optimal_stake(self, win_probability):
        """
        Calculate optimal stake using enhanced strategies to maximize profits
        while managing risk effectively
        
        Args:
            win_probability (float): Estimated probability of winning (0.0-1.0)
            
        Returns:
            float: Optimal stake amount
        """
        # Sanity check the probability - more aggressive win probability range
        win_probability = max(0.2, min(0.95, win_probability))
        
        # Calculate effective win probability based on recent performance
        # This helps adjust for potential biases in the signal confidence
        if self.total_trades >= 10:
            # Blend the signal confidence with actual historical win rate
            historical_weight = min(0.7, self.total_trades / 50)  # Max 70% weight to historical data
            signal_weight = 1 - historical_weight
            
            effective_win_prob = (win_probability * signal_weight) + (self.win_rate * historical_weight)
            
            # If we have strong recent performance, trust the signals more
            if self.consecutive_wins >= 3:
                effective_win_prob = win_probability * 0.7 + effective_win_prob * 0.3
                
            win_probability = effective_win_prob
        
        # Payout ratio for binary options - optimized based on market data
        # Different markets may have different payout ratios
        base_payout_ratio = 0.85  # Assuming 85% typical payout
        
        # Adjust payout ratio based on market conditions and volatility
        volatility_adjustment = 0.05 if hasattr(self, 'market_volatility') and self.market_volatility == 'high' else 0
        time_of_day_adjustment = 0.02 if hasattr(self, 'favorable_time') and self.favorable_time else 0
        
        # Total adjusted payout ratio
        payout_ratio = base_payout_ratio + volatility_adjustment + time_of_day_adjustment
        
        # Edge calculation with more accurate formula
        edge = (win_probability * (1 + payout_ratio)) - 1
        
        # ---- STAKE CALCULATION STRATEGIES ----
        
        # Strategy 1: Enhanced Kelly Criterion (for positive edge situations)
        if edge > 0:
            # Kelly formula: f* = (bp - q) / b
            # where f* is optimal fraction, b is net odds, p is win probability, q is loss probability
            kelly_fraction = (payout_ratio * win_probability - (1 - win_probability)) / payout_ratio
            
            # Apply safety factor - adjust based on confidence in edge
            # More aggressive when win probability is higher
            if win_probability > 0.75:
                safety_factor = self.config['RISK_MANAGEMENT']['kelly_fraction'] * 1.2  # More aggressive
            elif win_probability < 0.55:
                safety_factor = self.config['RISK_MANAGEMENT']['kelly_fraction'] * 0.7  # More conservative
            else:
                safety_factor = self.config['RISK_MANAGEMENT']['kelly_fraction']
                
            kelly_fraction *= safety_factor
            
            # Calculate effective bankroll - adapt to accumulated profit/loss
            # This creates a dynamic bankroll size that grows with profits
            base_bankroll = 100 * self.base_stake
            profit_adjustment = max(0, self.daily_profit * 0.5)  # 50% of accumulated profits
            
            effective_bankroll = base_bankroll + profit_adjustment
            kelly_stake = kelly_fraction * effective_bankroll
            
            # Strategy selection based on best past performance
            if self.consecutive_wins >= 3 and self.win_rate > 0.55:
                self.logger.info("Using progressive staking due to win streak")
                stake_strategy = "progressive"
            else:
                stake_strategy = "kelly"
                
        # Strategy 2: Minimum stake for neutral or negative edge
        else:
            kelly_stake = self.base_stake
            stake_strategy = "minimum"
        
        # ---- STAKE ADJUSTMENTS ----
        
        # Progressive stake adjustment for winning streaks
        if stake_strategy == "progressive" and self.consecutive_wins >= 2:
            # More aggressive progression formula
            base_increase = 0.15  # 15% base increase per win
            cap_factor = 2.0  # Cap at 2x increase
            
            streak_factor = 1.0 + (base_increase * min(self.consecutive_wins - 1, (cap_factor-1)/base_increase))
            kelly_stake *= streak_factor
            self.logger.info(f"Win streak of {self.consecutive_wins}: increasing stake by factor {streak_factor:.2f}")
        
        # Recovery mode adjustment
        if self.recovery_mode:
            if self.daily_profit < 0 and abs(self.daily_profit) > 10 * self.base_stake:
                # Deep recovery - use more conservative approach
                recovery_factor = self.config['RISK_MANAGEMENT']['recovery_factor'] * 0.7
                self.logger.info(f"Deep recovery mode: reducing stake significantly (factor: {recovery_factor:.2f})")
            else:
                # Standard recovery
                recovery_factor = self.config['RISK_MANAGEMENT']['recovery_factor']
                self.logger.info(f"Recovery mode: adjusting stake (factor: {recovery_factor:.2f})")
                
            kelly_stake *= recovery_factor
        
        # Special adjustment for very high win probability
        if win_probability > 0.8 and edge > 0.2 and not self.recovery_mode:
            high_prob_bonus = 1.2  # 20% bonus for very high probability trades
            kelly_stake *= high_prob_bonus
            self.logger.info(f"High probability trade bonus: {high_prob_bonus:.2f}x")
        
        # Time-based adjustment - reduce stakes during low-win periods if detected
        if hasattr(self, 'current_hour_win_rate') and self.current_hour_win_rate < 0.45:
            time_penalty = 0.7  # Reduce stakes by 30% during historically poor performance hours
            kelly_stake *= time_penalty
            self.logger.info(f"Poor historical time period: reducing stake by {(1-time_penalty)*100:.0f}%")
        
        # Ensure stake is within limits - with dynamic upper limit based on success
        if self.win_rate > 0.65 and self.total_trades > 20:
            # Allow higher maximum stakes for proven strategies
            dynamic_max = self.max_stake * 1.25
        else:
            dynamic_max = self.max_stake
            
        stake = max(self.base_stake, min(kelly_stake, dynamic_max))
        
        # Round to 2 decimal places
        stake = round(stake, 2)
        
        self.logger.info(f"Calculated stake: ${stake} using {stake_strategy} strategy (win probability: {win_probability:.2f})")
        return stake
    
    def update_stake_with_martingale(self):
        """
        Update stake using a modified martingale approach after losses
        
        Returns:
            float: New stake amount
        """
        if self.consecutive_losses == 0:
            # Reset to base stake after a win
            new_stake = self.base_stake
        else:
            # Progressive stake increase, but with a more conservative factor than classic martingale
            # Instead of 2x, we use a factor of 1.5 with an additional damping as losses increase
            factor = 1.5
            damping = max(0.7, 1.0 - (self.consecutive_losses * 0.05))  # Dampen the increase as losses mount
            
            new_stake = self.base_stake * (factor ** self.consecutive_losses) * damping
            
            # Ensure stake is within limits
            new_stake = min(new_stake, self.max_stake)
        
        # Check if we've reached the max losses limit
        if self.consecutive_losses >= self.config['RISK_MANAGEMENT']['max_consecutive_losses']:
            self.recovery_mode = True
            self.logger.warning(f"Hit {self.consecutive_losses} consecutive losses - activating recovery mode")
            new_stake = self.base_stake  # Reset to base stake in extreme loss streaks
        
        self.logger.info(f"Updated stake using modified martingale: ${new_stake:.2f}")
        return round(new_stake, 2)
    
    def update_after_trade(self, won, profit, stake):
        """
        Update risk management stats after a trade
        
        Args:
            won (bool): Whether the trade was won
            profit (float): Profit amount (positive for win, negative for loss)
            stake (float): Stake amount
        """
        self.total_trades += 1
        self.total_invested += stake
        
        if won:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
            self.total_payout += stake + profit
            
            # Check if we should exit recovery mode
            if self.recovery_mode and self.consecutive_wins >= 2:
                self.recovery_mode = False
                self.logger.info("Exiting recovery mode after consecutive wins")
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
            
            # Enter recovery mode after significant consecutive losses
            if self.consecutive_losses >= 3:
                self.recovery_mode = True
                self.logger.warning(f"Entering recovery mode after {self.consecutive_losses} consecutive losses")
        
        # Update win rate
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.5
        
        # Update daily profit
        self.daily_profit += profit
        
        # Record trade
        self.trade_history.append({
            'timestamp': datetime.now(),
            'won': won,
            'profit': profit,
            'stake': stake,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'daily_profit': self.daily_profit
        })
        
        # Check if daily loss limit reached
        if self.daily_profit <= -self.max_daily_loss:
            self.daily_loss_limit_reached = True
            self.logger.warning(f"Daily loss limit of ${self.max_daily_loss} reached. Stopping trading.")
        
        # Check if profit target reached
        if self.daily_profit >= self.profit_target:
            self.profit_target_reached = True
            self.logger.info(f"Daily profit target of ${self.profit_target} reached!")
        
        # Log update
        result = "won" if won else "lost"
        self.logger.info(f"Trade {result}: profit=${profit:.2f}, daily profit=${self.daily_profit:.2f}, win rate={self.win_rate:.2f}")
    
    def should_continue_trading(self):
        """
        Determine if trading should continue
        
        Returns:
            bool: True if trading should continue, False otherwise
        """
        # Check if daily loss limit reached
        if self.daily_loss_limit_reached:
            self.logger.warning("Daily loss limit reached - stopping trading")
            return False
        
        # Check if profit target reached and we've made a significant number of trades
        if self.profit_target_reached and self.total_trades > 10:
            self.logger.info("Profit target reached - can stop trading to secure profits")
            # We could return False here, but for now we'll continue with a warning
        
        # Check if max trades reached
        if self.total_trades >= self.config['TRADING']['max_daily_trades']:
            self.logger.warning("Maximum daily trades reached - stopping trading")
            return False
        
        # Check if session timeout reached
        session_duration = datetime.now() - self.session_start
        if session_duration.total_seconds() >= self.config['TRADING']['session_timeout']:
            self.logger.warning("Session timeout reached - stopping trading")
            return False
        
        return True
    
    def get_session_summary(self):
        """
        Get a summary of the current trading session
        
        Returns:
            dict: Session statistics
        """
        session_duration = datetime.now() - self.session_start
        
        return {
            'session_start': self.session_start.strftime('%Y-%m-%d %H:%M:%S'),
            'session_duration': str(session_duration).split('.')[0],  # Remove microseconds
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate * 100, 2),
            'daily_profit': round(self.daily_profit, 2),
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'recovery_mode': self.recovery_mode,
            'profit_target_reached': self.profit_target_reached,
            'daily_loss_limit_reached': self.daily_loss_limit_reached
        }
