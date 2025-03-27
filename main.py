import os
import threading
import time
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from datetime import datetime
import logging

# Import bot components
from enhanced_binary_bot import EnhancedBinaryOptionsBot
import config

# Set up Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global bot instance
bot_instance = None
bot_thread = None
bot_status = {
    "running": False,
    "last_update": None,
    "trades_executed": 0,
    "win_count": 0,
    "loss_count": 0,
    "current_profit": 0,
    "current_market": None,
    "messages": []
}

def add_status_message(message, level="info"):
    """Add a message to the status messages queue"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    bot_status["messages"].insert(0, {
        "timestamp": timestamp,
        "message": message,
        "level": level
    })
    # Keep only the last 100 messages
    if len(bot_status["messages"]) > 100:
        bot_status["messages"] = bot_status["messages"][:100]
    bot_status["last_update"] = datetime.now()
    logger.info(f"Bot status: {message}")

def bot_monitor_thread():
    """Thread to monitor bot status and update statistics"""
    global bot_instance, bot_status
    
    while bot_status["running"]:
        if bot_instance:
            try:
                # Update status from the running bot
                bot_status["current_market"] = bot_instance.current_market
                
                if hasattr(bot_instance, 'risk_manager') and bot_instance.risk_manager:
                    stats = bot_instance.risk_manager.get_session_summary()
                    bot_status["trades_executed"] = stats.get('total_trades', 0)
                    bot_status["win_count"] = stats.get('winning_trades', 0)
                    bot_status["loss_count"] = stats.get('losing_trades', 0)
                    bot_status["current_profit"] = stats.get('daily_profit', 0)
                
                # Add some messages about current state
                if bot_instance.current_market and bot_status["current_market"] != bot_instance.current_market:
                    add_status_message(f"Trading on market: {bot_instance.current_market}")
                
                bot_status["last_update"] = datetime.now()
            except Exception as e:
                add_status_message(f"Error updating bot status: {str(e)}", "error")
        
        # Check every 5 seconds
        time.sleep(5)

def run_bot_in_thread(api_token):
    """Run the bot in a separate thread"""
    global bot_instance, bot_status
    
    try:
        # Initialize bot
        add_status_message("Initializing Enhanced Binary Options Bot...")
        bot_instance = EnhancedBinaryOptionsBot(api_token=api_token)
        
        # Custom logger to capture bot messages
        class UIHandler(logging.Handler):
            def emit(self, record):
                log_entry = self.format(record)
                level = record.levelname.lower()
                add_status_message(log_entry, level)
        
        # Add UI handler to bot logger
        ui_handler = UIHandler()
        ui_handler.setFormatter(logging.Formatter('%(message)s'))
        bot_instance.logger.addHandler(ui_handler)
        
        # Start the bot
        add_status_message("Starting bot...")
        bot_instance.run()
    except Exception as e:
        add_status_message(f"Error starting bot: {str(e)}", "error")
        bot_status["running"] = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_bot', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    global bot_thread, bot_status
    
    if bot_status["running"]:
        return jsonify({"success": False, "message": "Bot is already running"})
    
    api_token = request.form.get('api_token')
    if not api_token:
        api_token = os.environ.get('DERIV_API_TOKEN')
        if not api_token:
            return jsonify({"success": False, "message": "Please provide an API token"})
    
    # Reset status
    bot_status = {
        "running": True,
        "last_update": datetime.now(),
        "trades_executed": 0,
        "win_count": 0,
        "loss_count": 0,
        "current_profit": 0,
        "current_market": None,
        "messages": []
    }
    
    # Start bot thread
    bot_thread = threading.Thread(target=run_bot_in_thread, args=(api_token,))
    bot_thread.daemon = True
    bot_thread.start()
    
    # Start monitor thread
    monitor_thread = threading.Thread(target=bot_monitor_thread)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    return jsonify({"success": True, "message": "Bot started successfully"})

@app.route('/stop_bot', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    global bot_instance, bot_status
    
    if not bot_status["running"]:
        return jsonify({"success": False, "message": "Bot is not running"})
    
    try:
        # Signal the bot to stop
        bot_status["running"] = False
        if bot_instance and hasattr(bot_instance, 'ws') and bot_instance.ws:
            bot_instance.ws.close()
            add_status_message("Bot stopped", "warning")
        
        return jsonify({"success": True, "message": "Bot stopped successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error stopping bot: {str(e)}"})

@app.route('/bot_status')
def get_bot_status():
    """Get the current status of the bot"""
    global bot_status
    
    return jsonify(bot_status)

@app.route('/config')
def bot_config():
    """View and edit bot configuration"""
    markets = config.MARKETS
    risk_config = {
        'base_stake': config.RISK_MANAGEMENT['base_stake'],
        'max_stake': config.RISK_MANAGEMENT['max_stake'],
        'kelly_fraction': config.RISK_MANAGEMENT['kelly_fraction'],
        'max_daily_loss': config.TRADING['max_daily_loss'],
        'profit_target': config.TRADING['profit_target']
    }
    return render_template('config.html', markets=markets, risk_config=risk_config)

# Add a route to serve the instructions page
@app.route('/instructions')
def instructions():
    return render_template('instructions.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)