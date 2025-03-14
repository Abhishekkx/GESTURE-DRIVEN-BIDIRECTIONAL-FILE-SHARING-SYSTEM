import socket
import os
import cv2
import mediapipe as mp
import threading
import queue
import time
import datetime
import logging
import asyncio
import websockets
import json
import glob
from mss import mss

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server_log.txt'),
        logging.StreamHandler()
    ]
)

# Network Configuration
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 12345
WEBSOCKET_PORT = 8765

# Initialize MediaPipe Hands with robust configuration
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class FileTransferServer:
    def __init__(self, host, port):
        """ Initialize the FileTransferServer with enhanced connection handling """
        self.host = host
        self.port = port
        self.server_socket = None
        self.connection = None
        self.client_address = None
        self.file_queue = queue.Queue()
        self.is_running = False
        self.connection_lock = threading.Lock()
        self.connection_timeout = 30  # Increased timeout
        self.reconnect_delay = 5  # Delay between reconnection attempts

    def start_server(self):
        """ Robust server socket initialization """
        try:
            # Create IPv4 TCP socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # Enable socket reuse
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # Bind to all interfaces
            self.server_socket.bind((self.host, self.port))

            # Listen with a backlog of 5 connections
            self.server_socket.listen(5)

            # Set a timeout for accept operations
            self.server_socket.settimeout(self.connection_timeout)

            logging.info(f"File Transfer Server started on {self.host}:{self.port}")
            self.is_running = True
            return True

        except Exception as e:
            logging.critical(f"Failed to start server: {e}")
            self.is_running = False
            return False

    def accept_connection(self):
        """ Enhanced connection acceptance with error handling """
        try:
            # Use a lock to prevent concurrent connection attempts
            with self.connection_lock:
                # Close existing connection if any
                if self.connection:
                    try:
                        self.connection.close()
                    except:
                        pass

                # Accept new connection with timeout
                logging.info("Waiting for client connection...")
                self.connection, self.client_address = self.server_socket.accept()

                # Configure socket options
                self.connection.settimeout(self.connection_timeout)
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                logging.info(f"Connected to client: {self.client_address}")
                return True

        except socket.timeout:
            logging.warning("Connection attempt timed out")
            return False

        except Exception as e:
            logging.error(f"Connection error: {e}")
            # Wait before attempting to reconnect
            time.sleep(self.reconnect_delay)
            return False

    def send_file(self, filepath):
        """ Robust file sending method """
        try:
            # Validate file existence
            if not os.path.exists(filepath):
                logging.error(f"File not found: {filepath}")
                return False

            # Ensure active connection
            if not self.connection:
                if not self.accept_connection():
                    return False

            # Prepare file metadata
            filename = os.path.basename(filepath)
            filesize = os.path.getsize(filepath)

            # Send filename length
            self.connection.sendall(len(filename).to_bytes(4, 'big'))
            # Send filename
            self.connection.sendall(filename.encode('utf-8'))
            # Send filesize
            self.connection.sendall(filesize.to_bytes(8, 'big'))

            # Send file contents in chunks
            with open(filepath, 'rb') as f:
                while chunk := f.read(4096):
                    self.connection.sendall(chunk)

            logging.info(f"Successfully sent file: {filepath}")
            return True

        except (ConnectionResetError, BrokenPipeError):
            logging.warning("Connection lost during file transfer")
            self.connection = None
            return False

        except Exception as e:
            logging.error(f"File sending error: {e}")
            return False

    def receive_file(self):
        """ Robust file receiving method """
        try:
            # Ensure active connection
            if not self.connection:
                if not self.accept_connection():
                    return None

            # Receive filename length
            filename_length = int.from_bytes(self.connection.recv(4), 'big')

            # Receive filename
            filename = self.connection.recv(filename_length).decode('utf-8')
            filepath = os.path.join('downloads', filename)

            # Receive filesize
            filesize = int.from_bytes(self.connection.recv(8), 'big')

            # Receive file contents
            with open(filepath, 'wb') as f:
                bytes_received = 0
                while bytes_received < filesize:
                    chunk = self.connection.recv(min(4096, filesize - bytes_received))
                    if not chunk:
                        logging.warning("Connection lost during file reception")
                        return None
                    f.write(chunk)
                    bytes_received += len(chunk)

            logging.info(f"File received: {filepath}")
            return filepath

        except Exception as e:
            logging.error(f"File receiving error: {e}")
            return None

    def file_transfer_thread(self):
        """ Background thread for file transfer operations """
        while self.is_running:
            try:
                # Ensure connection is established
                if not self.connection:
                    self.accept_connection()

                # Process file queue
                if not self.file_queue.empty():
                    filepath = self.file_queue.get()
                    self.send_file(filepath)

                time.sleep(0.1)

            except Exception as e:
                logging.error(f"Transfer thread error: {e}")
                time.sleep(1)

# Rest of the existing functions remain the same...

def main():
    try:
        # Create download directories
        create_download_directory()

        # Initialize file transfer server
        file_server = FileTransferServer(HOST, PORT)
        if not file_server.start_server():
            logging.critical("Failed to start file transfer server")
            return

        # Start file transfer thread
        transfer_thread = threading.Thread(target=file_server.file_transfer_thread, daemon=True)
        transfer_thread.start()

        # Open webcam (changed to 0)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.critical("Could not open webcam!")
            return

        # Set webcam resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        screenshot_count = 0
        previous_state = None
        state_change_time = time.time()

        # Simultaneously run WebSocket server
        websocket_thread = threading.Thread(target=lambda: asyncio.run(main_async()), daemon=True)
        websocket_thread.start()

        # Main hand gesture recognition loop
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture frame")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            current_state = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    current_time = time.time()

                    if is_hand_closed(hand_landmarks):
                        current_state = 'closed'
                        if previous_state != 'closed':
                            state_change_time = current_time
                    elif is_hand_open(hand_landmarks):
                        current_state = 'open'
                        if previous_state != 'open':
                            state_change_time = current_time

                    # Screenshot on closed hand after open hand
                    if (current_state == 'closed' and previous_state == 'open' and 
                        current_time - state_change_time < 1.5):
                        screenshot_count += 1
                        screenshot_path = os.path.join('screenshots',
                                                       f'screenshot_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                        with mss() as sct:
                            sct.shot(output=screenshot_path)
                        file_server.file_queue.put(screenshot_path)
                        logging.info(f"Screenshot captured: {screenshot_path}")

                    # Receive file on open hand after closed hand
                    elif (current_state == 'open' and previous_state == 'closed' and 
                          current_time - state_change_time < 1.5):
                        received_file = file_server.receive_file()
                        if received_file:
                            logging.info(f"Received file: {received_file}")

                if current_state:
                    previous_state = current_state

            cv2.putText(frame, f"Screenshots: {screenshot_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Hand Gesture Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        file_server.is_running = False

    except Exception as e:
        logging.error(f"Exception in main(): {e}")
        import traceback
        traceback.print_exc()

def is_hand_closed(hand_landmarks):
    """ Detect if the hand is closed """
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    finger_mcps = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]
    return sum(1 for tip, mcp in zip(finger_tips, finger_mcps) if
               hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mcp].y) >= 3

def is_hand_open(hand_landmarks):
    """ Detect if the hand is open """
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    finger_mcps = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]
    return sum(1 for tip, mcp in zip(finger_tips, finger_mcps) if
               hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y) >= 3

def create_download_directory():
    """ Create necessary directories with comprehensive error handling """
    directories = ['downloads', 'screenshots']
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Created directory: {directory}")
        except PermissionError:
            logging.critical(f"Permission denied creating {directory}")
            raise
        except OSError as e:
            logging.error(f"Error creating {directory}: {e}")
            raise

async def main_async():
    # Start WebSocket server for real-time image streaming
    websocket_server = await start_websocket_server()
    await websocket_server.wait_closed()

def start_websocket_server():
    """ Start WebSocket server """
    websocket_server = websockets.serve(
        websocket_handler, 
        'localhost', 
        WEBSOCKET_PORT
    )
    return websocket_server

async def websocket_handler(websocket, path):
    try:
        while True:
            # Get current images
            images = await get_images()
            
            # Send images to the connected client
            await websocket.send(json.dumps(images))
            
            # Wait for some time before next update
            await asyncio.sleep(5)  # Update every 5 seconds
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def get_images():
    """
    Retrieve image files from the download directory
    Sort by modification time, most recent first
    """
    # Get all image files (adjust extensions as needed)
    image_files = glob.glob(os.path.join('downloads', '*.png')) + \
                  glob.glob(os.path.join('downloads', '*.jpg')) + \
                  glob.glob(os.path.join('downloads', '*.jpeg'))
    
    # Create image data with metadata
    images = []
    for file_path in sorted(image_files, key=os.path.getmtime, reverse=True):
        filename = os.path.basename(file_path)
        mod_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                 time.localtime(os.path.getmtime(file_path)))
        
        images.append({
            'filename': filename,
            'path': f'/downloads/{filename}',  # Web-accessible path
            'timestamp': mod_time
        })
    
    return images

if __name__ == "__main__":
    main()