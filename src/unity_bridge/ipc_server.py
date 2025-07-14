import socket, threading, json, signal, sys

HOST, PORT = '127.0.0.1', 5005

# Graceful shutdown flag
global_shutdown = False

# Handler to catch SIGINT and SIGTERM

def _shutdown(signum, frame):
    global global_shutdown
    print("\n[IPC] Shutdown signal received, stopping server...")
    global_shutdown = True

signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


def handle_client(conn):
    try:
        data = conn.recv(4096).decode()
        requests = json.loads(data)
        responses = {"results": f"Received {requests.get('method')}"}
        conn.sendall(json.dumps(responses).encode())
    except Exception as e:
        print(f"[IPC] Error handling client: {e}")
    finally:
        conn.close()


def serve():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    server_socket.settimeout(1.0)  # allow periodic shutdown checks

    print(f"[IPC] Listening on {HOST}:{PORT} (Ctrl+C to stop)")
    while not global_shutdown:
        try:
            conn, addr = server_socket.accept()
        except socket.timeout:
            continue
        except OSError:
            break

        print(f"[IPC] Accepted connection from {addr}")
        threading.Thread(target=handle_client, args=(conn,), daemon=True).start()

    server_socket.close()
    print("[IPC] Server has shut down.")


if __name__ == "__main__":
    serve()
