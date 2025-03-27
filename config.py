"""
Configuration settings for the Enhanced Binary Options Bot
"""

# API connection settings
API_SETTINGS = {
    "app_id": "1089",  # Deriv app ID
    "api_token": "8fRRApGnNy0TY6T",  # Default API token (can be overridden by environment variable)
    "websocket_url": "wss://ws.binaryws.com/websockets/v3"
}

# Available synthetic markets
MARKETS = {
    "volatility_indices": [
        "R_10", "R_25", "R_50", "R_75", "R_100",
        "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V"
    ]
}

# Trading settings
TRADING = {
    "default_duration": 1,   # Default duration for contracts
    "default_duration_unit": "t",  # t=ticks, s=seconds, m=minutes, h=hours, d=days
    "max_daily_loss": 50,    # Maximum allowed daily loss in USD
    "max_daily_trades": 100, # Maximum number of trades per day
    "profit_target": 25,     # Daily profit target in USD
    "session_timeout": 3600, # Session timeout in seconds (1 hour)
}

# Risk management settings
RISK_MANAGEMENT = {
    "base_stake": 1.0,       # Base stake amount in USD
    "max_stake": 20.0,       # Maximum stake amount in USD
    "kelly_fraction": 0.25,  # Kelly criterion fraction (conservative)
    "recovery_factor": 0.5,  # Factor for recovery mode stakes
    "max_consecutive_losses": 5,  # Maximum consecutive losses before strategy change
    "profit_lock_percentage": 0.5,  # Lock in 50% of profits
}

# Technical analysis settings
TECHNICAL_ANALYSIS = {
    "rsi_period": 14,        # Period for RSI calculation
    "rsi_overbought": 70,    # RSI overbought threshold
    "rsi_oversold": 30,      # RSI oversold threshold
    "macd_fast": 12,         # MACD fast period
    "macd_slow": 26,         # MACD slow period
    "macd_signal": 9,        # MACD signal period
    "bollinger_period": 20,  # Bollinger bands period
    "bollinger_std": 2,      # Bollinger bands standard deviation
    "min_data_points": 30,   # Minimum data points for analysis
}

# Logging settings
LOGGING = {
    "level": "INFO",         # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    "log_to_file": True,     # Whether to log to file
    "log_file": "enhanced_binary_bot.log", # Log file name
    "log_to_console": True,  # Whether to log to console
}
