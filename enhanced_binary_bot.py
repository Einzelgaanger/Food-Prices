#!/usr/bin/env python3
"""
Enhanced Binary Options Trading Bot with improved win-rate logic and risk management
"""
import websocket
import json
import threading
import time
import random
import logging
import sys
import os
from datetime import datetime

# Import custom modules
from technical_indicators import TechnicalIndicators
from risk_management import RiskManager
from market_analyzer import MarketAnalyzer
import config

class EnhancedBinaryOptionsBot:
    def __init__(self, api_token=None, app_id=None):
        """
        Initialize the enhanced binary options trading bot
        
        Args:
            api_token (str, optional): Deriv API token. Defaults to environment variable.
            app_id (str, optional): Deriv App ID. Defaults to environment variable.
        """
        # Set up logging
        self._setup_logging()
        
        # Get API credentials
        self.api_token = api_token or os.environ.get('DERIV_API_TOKEN', '8fRRApGnNy0TY6T')
        self.app_id = app_id or os.environ.get('DERIV_APP_ID', config.API_SETTINGS['app_id'])
        
        if not self.api_token:
            self.logger.critical("API token not provided. Please set DERIV_API_TOKEN environment variable.")
            sys.exit(1)
        else:
            self.logger.info(f"Using API token: {self.api_token}")
            
        # WebSocket connection
        self.websocket_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        self.ws = None
        self.ws_thread = None
        
        # Request ID counter
        self.req_id = 1
        
        # Bot state
        self.authorized = False
        self.active_markets = config.MARKETS['volatility_indices'].copy()
        self.current_market = None
        self.active_contract = None
        
        # Trading parameters
        self.trade_duration = config.TRADING['default_duration']
        self.trade_duration_unit = config.TRADING['default_duration_unit']
        
        # Trading state
        self.is_trading = False
        self.waiting_for_contract_settlement = False
        self.gathering_market_data = False
        self.market_data_collection_start = None
        
        # Initialize components
        self.technical_indicators = TechnicalIndicators(config.__dict__)
        self.risk_manager = RiskManager(config.__dict__)
        self.market_analyzer = MarketAnalyzer(config.__dict__)
        
        self.logger.info("Enhanced Binary Options Bot initialized")
    
    def _setup_logging(self):
        """Set up logging configuration"""
        log_level = getattr(logging, config.LOGGING['level'])
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        handlers = []
        if config.LOGGING['log_to_file']:
            handlers.append(logging.FileHandler(config.LOGGING['log_file']))
        if config.LOGGING['log_to_console']:
            handlers.append(logging.StreamHandler())
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=handlers
        )
        
        self.logger = logging.getLogger('enhanced_binary_bot')
    
    def get_next_req_id(self):
        """Get the next request ID and increment counter"""
        req_id = self.req_id
        self.req_id += 1
        return req_id
    
    def connect(self):
        """Establish WebSocket connection"""
        self.logger.info("Connecting to Deriv API...")
        self.ws = websocket.WebSocketApp(
            self.websocket_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def on_open(self, ws):
        """WebSocket connection opened"""
        self.logger.info("WebSocket connection opened")
        self.authorize()
    
    def on_message(self, ws, message):
        """Handle incoming messages"""
        data = json.loads(message)
        msg_type = data.get('msg_type')
        
        try:
            # Handle authorization
            if msg_type == 'authorize' and data.get('authorize'):
                self.authorized = True
                self.logger.info("Successfully authorized with Deriv API")
                # Start gathering market data
                self.start_gathering_market_data()
            
            # Handle tick updates
            elif msg_type == 'tick' and data.get('tick'):
                self.handle_tick(data['tick'])
            
            # Handle history response
            elif msg_type == 'history' and data.get('history'):
                self.handle_history(data['history'])
            
            # Handle buy contract response
            elif msg_type == 'buy' and data.get('buy'):
                self.handle_buy_response(data)
            
            # Handle contract updates
            elif msg_type == 'proposal_open_contract' and data.get('proposal_open_contract'):
                self.handle_contract_update(data['proposal_open_contract'])
            
            # Handle errors
            elif data.get('error'):
                error_message = data['error']['message']
                error_code = data.get('error', {}).get('code', 'unknown')
                self.logger.error(f"API Error: {error_message} (Code: {error_code})")
                self.logger.error(f"Request that caused error: {json.dumps(data.get('echo_req', {}))}")
                
                # Check if error is insufficient balance
                if "balance" in error_message.lower() and "insufficient" in error_message.lower():
                    self.logger.critical("Insufficient balance. Exiting program.")
                    self.ws.close()
                    sys.exit(1)
                
                # If market is closed, try the next market
                if "market is closed" in error_message.lower() or "market" in error_message.lower():
                    self.logger.warning(f"Market {self.current_market} is closed. Trying next market.")
                    self.select_market()
                    self.subscribe_to_market_data()
        
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}", exc_info=True)
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        self.logger.error(f"WebSocket Error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closure"""
        self.logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        # Attempt to reconnect after a brief pause
        time.sleep(5)
        self.connect()
    
    def authorize(self):
        """Authorize with the Deriv API"""
        req_id = self.get_next_req_id()
        self.ws.send(json.dumps({
            "authorize": self.api_token,
            "req_id": req_id
        }))
    
    def start_gathering_market_data(self):
        """Start gathering market data before trading"""
        self.logger.info("Starting market data collection phase")
        self.gathering_market_data = True
        self.market_data_collection_start = datetime.now()
        
        # Subscribe to all markets to gather data
        for market in self.active_markets:
            self.subscribe_to_market_ticks(market)
            
        # Set a timer to start trading after collecting data
        threading.Timer(120, self.finish_gathering_market_data).start()  # 2 minutes of data collection
    
    def finish_gathering_market_data(self):
        """Finish gathering market data and start trading"""
        if not self.gathering_market_data:
            return
            
        self.gathering_market_data = False
        duration = datetime.now() - self.market_data_collection_start
        self.logger.info(f"Finished market data collection phase ({duration.seconds} seconds)")
        
        # Rank markets based on collected data
        self.market_analyzer.rank_markets()
        
        # Select the best market and start trading
        self.select_market()
        self.start_trading()
    
    def subscribe_to_market_ticks(self, market):
        """
        Subscribe to market tick data
        
        Args:
            market (str): Market symbol
        """
        req_id = self.get_next_req_id()
        self.ws.send(json.dumps({
            "ticks": market,
            "subscribe": 1,
            "req_id": req_id
        }))
        self.logger.debug(f"Subscribed to ticks for {market}")
    
    def get_market_history(self, market, count=100):
        """
        Request historical tick data for a market
        
        Args:
            market (str): Market symbol
            count (int): Number of historical ticks to request
        """
        req_id = self.get_next_req_id()
        self.ws.send(json.dumps({
            "ticks_history": market,
            "count": count,
            "end": "latest",
            "style": "ticks",
            "req_id": req_id
        }))
        self.logger.debug(f"Requested historical ticks for {market}")
    
    def handle_tick(self, tick_data):
        """
        Handle tick data updates
        
        Args:
            tick_data (dict): Tick data
        """
        symbol = tick_data.get("symbol")
        if not symbol:
            return
            
        # Update market data
        self.market_analyzer.update_market_data(symbol, tick_data)
        
        # If this is for our current market, update technical indicators
        if symbol == self.current_market:
            price = tick_data.get("quote")
            if price and self.technical_indicators.add_price(price, tick_data):
                # If we're gathering data, don't trade yet
                if not self.gathering_market_data and not self.waiting_for_contract_settlement:
                    # Check if we should trade based on indicators
                    self.check_trading_opportunity()
    
    def handle_history(self, history_data):
        """
        Handle historical tick data
        
        Args:
            history_data (dict): Historical tick data
        """
        symbol = history_data.get("symbol")
        prices = history_data.get("prices", [])
        times = history_data.get("times", [])
        
        if not symbol or not prices or not times:
            self.logger.warning("Incomplete history data received")
            return
            
        self.logger.info(f"Received {len(prices)} historical ticks for {symbol}")
        
        # Process historical data
        for i in range(len(prices)):
            tick_data = {
                "symbol": symbol,
                "price": prices[i],
                "time": times[i]
            }
            self.market_analyzer.update_market_data(symbol, tick_data)
        
        # Update technical indicators with last price if this is our current market
        if symbol == self.current_market and prices:
            self.technical_indicators.add_price(prices[-1])
    
    def select_market(self):
        """Select the best market to trade on"""
        previous_market = self.current_market
        
        # Get best market from market analyzer
        self.current_market = self.market_analyzer.get_best_market()
        
        if previous_market != self.current_market:
            self.logger.info(f"Selected market: {self.current_market}")
            
            # Subscribe to market data if needed
            self.subscribe_to_market_data()
    
    def subscribe_to_market_data(self):
        """Subscribe to data for the current market"""
        if not self.current_market:
            self.logger.error("No market selected to subscribe to")
            return
            
        # Subscribe to ticks
        self.subscribe_to_market_ticks(self.current_market)
        
        # Get historical data
        self.get_market_history(self.current_market)
    
    def start_trading(self):
        """Start trading"""
        if not self.authorized or not self.current_market:
            self.logger.error("Not authorized or no market selected")
            return
        
        self.is_trading = True
        self.waiting_for_contract_settlement = False
        self.logger.info(f"Starting trading on {self.current_market}")
        
        # Use market analyzer to get optimal contract settings
        contract_settings = self.market_analyzer.get_ideal_contract_settings(self.current_market)
        self.trade_duration = contract_settings['duration']
        self.trade_duration_unit = contract_settings['duration_unit']
        
        self.logger.info(f"Contract settings: duration={self.trade_duration} {self.trade_duration_unit}")
        
        # Check for trading opportunity immediately
        self.check_trading_opportunity()
    
    def check_trading_opportunity(self):
        """
        Enhanced trading opportunity detection that combines technical indicators,
        market analysis and risk management to identify high-probability trades
        """
        if not self.is_trading or self.waiting_for_contract_settlement:
            return
        
        # Get trading signal from technical indicators
        signal, confidence, reason = self.technical_indicators.get_trade_signal()
        
        # Get market conditions from the market analyzer
        market_conditions = self.market_analyzer.get_current_market_conditions()
        
        # Share market information with risk manager for better stake calculation
        if hasattr(self.risk_manager, 'market_volatility'):
            # Get volatility from technical indicators
            self.risk_manager.market_volatility = self.technical_indicators.indicators['volatility']
            
        if hasattr(self.risk_manager, 'favorable_time'):
            # Pass time favorability information to risk manager
            self.risk_manager.favorable_time = market_conditions['is_favorable_time']
            
        if hasattr(self.risk_manager, 'current_hour_win_rate'):
            # Pass current hour win rate to risk manager
            self.risk_manager.current_hour_win_rate = market_conditions['current_hour_win_rate']
        
        # Enhance confidence based on market conditions
        confidence_adjustment = 0.0
        
        # If trading during favorable hours, boost confidence
        if market_conditions['is_favorable_time']:
            confidence_adjustment += 0.05
            self.logger.info(f"Trading during favorable hour (win rate: {market_conditions['current_hour_win_rate']:.2f})")
        
        # If recently switched to a better market, increase confidence
        if self.current_market in market_conditions['best_markets'][:2]:
            confidence_adjustment += 0.03
            self.logger.info(f"Trading on top-ranked market: {self.current_market}")
        
        # Apply the adjustment
        adjusted_confidence = min(1.0, confidence + confidence_adjustment)
        
        # Enhanced decision logic for trading
        should_trade = False
        min_confidence = 0.65  # Increased minimum confidence threshold
        
        # Only trade when we have clear signals with good confidence
        if signal != 'wait' and adjusted_confidence >= min_confidence:
            # Additional verification: check if signal aligns with historical performance
            market_win_rate = self.market_analyzer.get_win_rate(self.current_market)
            if market_win_rate > 0.5:  # Market has positive historical performance
                should_trade = True
            elif adjusted_confidence > 0.75:  # Very strong signal can override negative history
                should_trade = True
                self.logger.info("Strong signal overriding negative market history")
            else:
                self.logger.info(f"Skipping trade - market has poor win rate: {market_win_rate:.2f}")
        
        # Execute trade if conditions are met
        if should_trade:
            self.logger.info(f"High-probability trading opportunity detected: {signal} with confidence {adjusted_confidence:.2f} (base: {confidence:.2f}, adjustment: +{confidence_adjustment:.2f})")
            self.logger.info(f"Signal reason: {reason}")
            self.place_trade(signal, adjusted_confidence)
        else:
            self.logger.debug(f"No clear trading opportunity. Signal: {signal}, Confidence: {adjusted_confidence:.2f}")
            
            # Consider market switching if consistently not finding opportunities
            if random.random() < 0.2:  # 20% chance to re-evaluate markets
                top_markets = self.market_analyzer.rank_markets()
                if self.current_market not in top_markets[:3]:
                    self.logger.info(f"Current market {self.current_market} not performing well. Switching markets.")
                    self.select_market()
                    self.subscribe_to_market_data()
            
            # Check again after a short delay
            threading.Timer(5, self.check_trading_opportunity).start()
    
    def place_trade(self, contract_type, confidence):
        """
        Place a binary options trade
        
        Args:
            contract_type (str): Contract type (CALL or PUT)
            confidence (float): Signal confidence (0.0-1.0)
        """
        if not self.authorized or not self.current_market:
            self.logger.error("Cannot place trade: Not authorized or no market selected")
            return
            
        # Check if trading should continue
        if not self.risk_manager.should_continue_trading():
            self.logger.warning("Risk management indicates trading should stop")
            self.is_trading = False
            return
        
        # Calculate stake amount based on confidence and win probability
        stake = self.risk_manager.calculate_optimal_stake(confidence)
        
        # Log trade details
        self.logger.info(f"Placing {contract_type} trade on {self.current_market}")
        self.logger.info(f"Stake: ${stake}")
        self.logger.info(f"Contract duration: {self.trade_duration} {self.trade_duration_unit}")
        
        # Get a new request ID
        req_id = self.get_next_req_id()
        
        # Send trade request
        self.ws.send(json.dumps({
            "buy": 1,
            "price": stake,
            "parameters": {
                "amount": stake,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": self.trade_duration,
                "duration_unit": self.trade_duration_unit,
                "symbol": self.current_market
            },
            "req_id": req_id
        }))
        
        self.waiting_for_contract_settlement = True
    
    def handle_buy_response(self, data):
        """
        Handle the response after placing a contract
        
        Args:
            data (dict): Buy response data
        """
        buy_data = data.get('buy')
        req_id = data.get('req_id')
        
        if buy_data:
            contract_id = buy_data.get("contract_id")
            self.active_contract = {
                "contract_id": contract_id,
                "type": buy_data.get("longcode", "").split(" ")[1],  # Extract CALL/PUT from longcode
                "stake": buy_data.get("buy_price"),
                "req_id": req_id
            }
            self.logger.info(f"Contract placed successfully. ID: {contract_id}")
            
            # Subscribe to contract updates
            subscription_req_id = self.get_next_req_id()
            self.ws.send(json.dumps({
                "proposal_open_contract": 1,
                "contract_id": contract_id,
                "subscribe": 1,
                "req_id": subscription_req_id
            }))
        else:
            self.logger.error(f"Failed to place contract. Request ID: {req_id}")
            self.waiting_for_contract_settlement = False
            
            # Move to the next market if there was an error
            self.select_market()
            self.subscribe_to_market_data()
            
            # Try again after a delay
            threading.Timer(5, self.check_trading_opportunity).start()
    
    def handle_contract_update(self, contract_data):
        """
        Handle updates for an open contract
        
        Args:
            contract_data (dict): Contract update data
        """
        if not self.active_contract:
            return
            
        contract_id = contract_data.get("contract_id")
        
        # Make sure this is our active contract
        if contract_id != self.active_contract["contract_id"]:
            return
            
        status = contract_data.get("status")
        
        if status == "open":
            # Contract still running
            return
            
        self.waiting_for_contract_settlement = False
        
        # Extract profit and other details
        profit = contract_data.get("profit", 0)
        stake = contract_data.get("buy_price", self.active_contract.get("stake", 0))
        contract_type = self.active_contract.get("type")
        
        if status == "won":
            self.logger.info(f"Contract won! Profit: ${profit}")
            
            # Update risk manager and market analyzer
            self.risk_manager.update_after_trade(True, profit, stake)
            self.market_analyzer.update_win_loss(self.current_market, True)
            
            # Log session summary
            session_summary = self.risk_manager.get_session_summary()
            self.logger.info(f"Session summary: Win rate: {session_summary['win_rate']}%, Profit: ${session_summary['daily_profit']}")
            
            # Check if we should continue trading
            if self.risk_manager.should_continue_trading():
                # Select market again and continue trading
                self.select_market()
                self.subscribe_to_market_data()
                
                # Continue trading after a short delay
                threading.Timer(5, self.check_trading_opportunity).start()
            else:
                self.logger.info("Stopping trading due to risk management constraints")
                self.is_trading = False
                
        elif status == "lost":
            self.logger.info(f"Contract lost. Loss: ${profit}")
            
            # Update risk manager and market analyzer
            self.risk_manager.update_after_trade(False, profit, stake)
            self.market_analyzer.update_win_loss(self.current_market, False)
            
            # Log session summary
            session_summary = self.risk_manager.get_session_summary()
            self.logger.info(f"Session summary: Win rate: {session_summary['win_rate']}%, Profit: ${session_summary['daily_profit']}")
            
            # Check if we should continue trading
            if self.risk_manager.should_continue_trading():
                # Select market again and continue trading
                self.select_market()
                self.subscribe_to_market_data()
                
                # Continue trading after a short delay
                threading.Timer(5, self.check_trading_opportunity).start()
            else:
                self.logger.info("Stopping trading due to risk management constraints")
                self.is_trading = False
        
        # Clear active contract
        self.active_contract = None
    
    def run(self):
        """Main bot loop"""
        self.logger.info("Starting Enhanced Binary Options Trading Bot")
        
        # Log available markets
        self.logger.info("Available synthetic indices markets:")
        for market_type, markets in config.MARKETS.items():
            self.logger.info(f"- {market_type}: {', '.join(markets)}")
        
        # Connect to API
        self.connect()
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
                
                # Check if WebSocket is still connected
                if not self.ws_thread.is_alive():
                    self.logger.warning("WebSocket thread died, reconnecting...")
                    self.connect()
                
                # Perform periodic market analysis if trading
                if self.is_trading and not self.waiting_for_contract_settlement:
                    # Every minute, re-rank markets to find better opportunities
                    if datetime.now().second == 0:
                        self.market_analyzer.rank_markets()
                
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received, shutting down...")
            if self.ws:
                self.ws.close()
            sys.exit(0)

if __name__ == "__main__":
    # Check for API token in environment
    api_token = os.environ.get('DERIV_API_TOKEN')
    app_id = os.environ.get('DERIV_APP_ID', config.API_SETTINGS['app_id'])
    
    if not api_token:
        print("Please set the DERIV_API_TOKEN environment variable")
        sys.exit(1)
    
    # Create and run the bot
    bot = EnhancedBinaryOptionsBot(api_token, app_id)
    bot.run()
