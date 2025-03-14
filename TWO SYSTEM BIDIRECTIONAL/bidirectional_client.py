import socket
import os
import cv2
import mediapipe as mp
import pyautogui
from datetime import datetime
import threading
import queue
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')

# Server Configuration
HOST = '192.168.0.103'  # Localhost for testing, change as needed
PORT = 12345

# Initialize MediaPipe Hands with more robust configuration
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Create directories with error handling
def create_directories():
    directories = ['screenshots', 'downloads']
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            logging.error(f"Could not create directory {directory}: {e}")

class FileTransferClient:
    def __init__(self, host, port):
        """
        Initialize the FileTransferClient with host and port.
        
        :param host: Server host address
        :param port: Server port number
        """
        self.host = host
        self.port = port
        self.connection = None
        self.file_queue = queue.Queue()
        self.is_running = False
        self.connection_lock = threading.Lock()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

    def connect(self):
        """Establish a connection to the server with exponential backoff"""
        try:
            with self.connection_lock:
                if self.connection:
                    self.connection.close()
                
                self.connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.connection.settimeout(10)  # Increased timeout
                self.connection.connect((self.host, self.port))
                logging.info("Connected to server successfully!")
                self.reconnect_attempts = 0
                return True
        except Exception as e:
            self.reconnect_attempts += 1
            logging.error(f"Connection failed (Attempt {self.reconnect_attempts}): {e}")
            
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                logging.critical("Max reconnection attempts reached. Stopping connection attempts.")
                return False
            
            time.sleep(2 ** self.reconnect_attempts)  # Exponential backoff
            return False

    def send_file(self, filepath):
        """Send a file to the server"""
        try:
            with self.connection_lock:
                if not self.connection:
                    self.connect()
                    if not self.connection:
                        return False

                # Get file details
                filename = os.path.basename(filepath)
                filesize = os.path.getsize(filepath)
                
                # Extended file transfer protocol
                filename_length = len(filename)
                self.connection.sendall(filename_length.to_bytes(4, byteorder='big'))
                self.connection.sendall(filename.encode('utf-8'))
                self.connection.sendall(filesize.to_bytes(8, byteorder='big'))
                
                logging.info(f"Sending file: {filename}")
                
                with open(filepath, 'rb') as f:
                    while True:
                        chunk = f.read(4096)
                        if not chunk:
                            break
                        self.connection.sendall(chunk)
                
                logging.info(f"File {filename} sent successfully!")
                return True
        
        except Exception as e:
            logging.error(f"File send error: {e}")
            return False

    def receive_file(self):
        """Receive a file from the server"""
        try:
            with self.connection_lock:
                if not self.connection:
                    self.connect()
                    if not self.connection:
                        return None

                # Receive file details
                filename_length_bytes = self.connection.recv(4)
                filename_length = int.from_bytes(filename_length_bytes, byteorder='big')

                filename_bytes = self.connection.recv(filename_length)
                current_filename = filename_bytes.decode('utf-8')

                current_filepath = os.path.join('downloads', current_filename)

                filesize_bytes = self.connection.recv(8)
                current_filesize = int.from_bytes(filesize_bytes, byteorder='big')

                logging.info(f"Receiving file: {current_filename}")
                with open(current_filepath, "wb") as f:
                    received_bytes = 0
                    while received_bytes < current_filesize:
                        chunk_size = min(4096, current_filesize - received_bytes)
                        data = self.connection.recv(chunk_size)
                        if not data:
                            break
                        f.write(data)
                        received_bytes += len(data)

                logging.info(f"File {current_filename} received successfully!")
                return current_filepath
        except Exception as e:
            logging.error(f"File receive error: {e}")
            return None

    def file_transfer_thread(self):
        """Background thread for file transfer operations"""
        while self.is_running:
            try:
                if not self.connection:
                    self.connect()
                
                if not self.file_queue.empty():
                    filepath = self.file_queue.get()
                    self.send_file(filepath)
                
                time.sleep(0.1)  # Prevent tight loop
            except Exception as e:
                logging.error(f"File transfer thread error: {e}")
                time.sleep(1)

def is_hand_closed(hand_landmarks):
    """Improved hand closure detection"""
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
    
    # More robust closed hand detection
    curled_count = sum(
        1 for tip, mcp in zip(finger_tips, finger_mcps)
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mcp].y
    )
    
    return curled_count >= 3

def is_hand_open(hand_landmarks):
    """Improved hand open detection"""
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
    
    # More robust open hand detection
    extended_count = sum(
        1 for tip, mcp in zip(finger_tips, finger_mcps)
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y
    )
    
    return extended_count >= 3

def main():
    # Create necessary directories
    create_directories()

    # Initialize file transfer client - NOW WITH ARGUMENTS
    file_client = FileTransferClient(HOST, PORT)
    file_client.is_running = True

    # Start file transfer thread
    transfer_thread = threading.Thread(target=file_client.file_transfer_thread, daemon=True)
    transfer_thread.start()

    # Open webcam with error handling
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.critical("Could not open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    screenshot_count = 0
    previous_state = None
    state_change_time = time.time()

    try:
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
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS
                    )
                    
                    current_time = time.time()
                    if is_hand_closed(hand_landmarks):
                        current_state = 'closed'
                        if previous_state != 'closed':
                            state_change_time = current_time
                    
                    elif is_hand_open(hand_landmarks):
                        current_state = 'open'
                        if previous_state != 'open':
                            state_change_time = current_time

                    # Take screenshot on closed hand after open hand
                    if (current_state == 'closed' and 
                        previous_state == 'open' and 
                        current_time - state_change_time < 1.5):
                        screenshot_count += 1
                        screenshot_path = os.path.join(
                            'screenshots', 
                            f'screenshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{screenshot_count}.png'
                        )
                        
                        screenshot = pyautogui.screenshot()
                        screenshot.save(screenshot_path)
                        file_client.file_queue.put(screenshot_path)
                        logging.info(f"Screenshot captured: {screenshot_path}")

                    # Receive file on open hand after closed hand
                    elif (current_state == 'open' and 
                          previous_state == 'closed' and 
                          current_time - state_change_time < 1.5):
                        received_file = file_client.receive_file()
                        if received_file:
                            logging.info(f"Received file: {received_file}")

            # Update previous state
            if current_state:
                previous_state = current_state
            
            cv2.putText(frame, f"Screenshots: {screenshot_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Bidirectional File Transfer", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logging.info("Stopping client...")
    finally:
        file_client.is_running = False
        transfer_thread.join()
        cap.release()
        cv2.destroyAllWindows()
        logging.info(f"Total screenshots taken: {screenshot_count}")

if __name__ == "__main__":
    main()