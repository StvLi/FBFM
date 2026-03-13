import websockets.sync.client
import time

uri = "ws://localhost:29056"
print(f"Testing WebSocket connection to {uri}...")

try:
    print("Attempting to connect...")
    conn = websockets.sync.client.connect(
        uri,
        compression=None,
        max_size=None,
        ping_interval=None,
        close_timeout=10,
        open_timeout=10,
        proxy=None  # 禁用代理
    )
    print("✅ WebSocket connection established!")
    
    print("Waiting for server message...")
    try:
        message = conn.recv()
        print(f"✅ Received message: {type(message)}, length: {len(message) if hasattr(message, '__len__') else 'N/A'}")
    except Exception as e:
        print(f"❌ Error receiving message: {e}")
        import traceback
        traceback.print_exc()
    
    conn.close()
    print("✅ Connection closed successfully")
    
except Exception as e:
    print(f"❌ WebSocket connection failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()