#!/usr/bin/env python3

# Try different imports
try:
    print("Trying to import websocket_client...")
    from websocket_client import WebSocketApp
    print("Successfully imported from websocket_client")
except ImportError:
    print("Could not import from websocket_client. Trying websocket._app...")
    try:
        from websocket._app import WebSocketApp
        print("Successfully imported from websocket._app")
    except ImportError:
        print("Could not import from websocket._app. Trying websocket directly...")
        try:
            import websocket
            WebSocketApp = websocket.WebSocketApp
            print("Successfully imported from websocket")
        except (ImportError, AttributeError) as e:
            print(f"Error: {e}")
            print("Let's check installed packages...")
            import subprocess
            print(subprocess.check_output(["pip", "list"]).decode())
import json
import os

def on_message(ws, message):
    print(f"Message received: {message}")
    
def on_error(ws, error):
    print(f"Error: {error}")
    
def on_close(ws, close_status_code, close_msg):
    print(f"Connection closed: {close_status_code} - {close_msg}")
    
def on_open(ws):
    print("Connection opened")
    api_token = os.environ.get('DERIV_API_TOKEN', '8fRRApGnNy0TY6T')
    ws.send(json.dumps({
        "authorize": api_token
    }))

if __name__ == "__main__":
    try:
        print("Testing websocket-client connection to Deriv API")
        app_id = "1089"  # Default app ID
        url = f"wss://ws.binaryws.com/websockets/v3?app_id={app_id}"
        
        ws = WebSocketApp(
            url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        print("Starting websocket connection...")
        ws.run_forever()
    except ImportError:
        print("Could not import WebSocketApp from websocket_client.")
        print("Let's try importing from websocket._app...")
        
        try:
            from websocket._app import WebSocketApp
            ws = WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            print("Successfully imported from websocket._app")
            ws.run_forever()
        except ImportError:
            print("Could not import WebSocketApp from websocket._app either.")
            print("Let's check what modules we have installed:")
            import sys
            print([module for module in sys.modules.keys() if 'websocket' in module])
            print("\nLet's try to import the websocket module and see what's in it:")
            import websocket
            print(dir(websocket))