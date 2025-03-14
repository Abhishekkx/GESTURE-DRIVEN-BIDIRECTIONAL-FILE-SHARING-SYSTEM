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
import json
from mss import mss
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)

class NetworkDiscovery:
    """Simple network discovery using UDP broadcast"""
    
    BROADCAST_PORT = 12345
    
    def __init__(self):
        self.peers = set()
        self.is_running = False
        
    def start_discovery(self):
        """Start peer discovery"""
        self.is_running = True
        
        # Start listening for broadcasts
        threading.Thread(target=self._listen_for_peers, daemon=True).start()
        # Start broadcasting presence
        threading.Thread(target=self._broadcast_presence, daemon=True).start()

    def _listen_for_peers(self):
        """Listen for peer announcements"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self.BROADCAST_PORT))
        
        while self.is_running:
            try:
                data, addr = sock.recvfrom(1024)
                if data == b'PEER_ANNOUNCE' and addr[0] not in self.peers:
                    self.peers.add(addr[0])
                    logging.info(f"Discovered peer: {addr[0]}")
            except Exception as e:
                logging.error(f"Discovery listen error: {e}")
                time.sleep(1)

    def _broadcast_presence(self):
        """Broadcast presence to network"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        while self.is_running:
            try:
                sock.sendto(b'PEER_ANNOUNCE', ('<broadcast>', self.BROADCAST_PORT))
            except Exception as e:
                logging.error(f"Broadcast error: {e}")
            time.sleep(5)

class P2PFileTransfer:
    def __init__(self, port=12346):
        self.discovery = NetworkDiscovery()
        self.peers = {}
        self.is_running = False
        self.file_queue = queue.Queue()
        self.connection_lock = threading.Lock()
        self.port = port
        
        # Initialize listening socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', self.port))

    def start(self):
        """Start P2P service"""
        self.is_running = True
        self.sock.listen(5)
        
        # Start discovery
        self.discovery.start_discovery()
        
        # Start connection handling
        threading.Thread(target=self.manage_connections, daemon=True).start()
        threading.Thread(target=self.process_file_queue, daemon=True).start()

    def stop(self):
        """Stop P2P service"""
        self.is_running = False
        self.discovery.is_running = False
        self.sock.close()
        for peer_sock in self.peers.values():
            peer_sock.close()

    def manage_connections(self):
        """Manage peer connections"""
        while self.is_running:
            # Try connecting to discovered peers
            for peer_addr in self.discovery.peers:
                if peer_addr not in self.peers:
                    self.connect_to_peer(peer_addr)
            
            # Accept incoming connections
            try:
                self.sock.settimeout(1)
                client_sock, address = self.sock.accept()
                if address[0] not in self.peers:
                    self.peers[address[0]] = client_sock
                    threading.Thread(
                        target=self.handle_peer_connection,
                        args=(client_sock, address[0]),
                        daemon=True
                    ).start()
            except socket.timeout:
                pass
            except Exception as e:
                if self.is_running:
                    logging.error(f"Connection accept error: {e}")
            
            time.sleep(1)

    def connect_to_peer(self, peer_addr):
        """Connect to a discovered peer"""
        try:
            peer_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            peer_sock.settimeout(5)
            peer_sock.connect((peer_addr, self.port))
            self.peers[peer_addr] = peer_sock
            threading.Thread(
                target=self.handle_peer_connection,
                args=(peer_sock, peer_addr),
                daemon=True
            ).start()
        except Exception as e:
            logging.debug(f"Could not connect to peer {peer_addr}: {e}")

    def handle_peer_connection(self, peer_sock, peer_addr):
        """Handle communication with a connected peer"""
        while self.is_running:
            try:
                # Receive file metadata
                metadata_size = int.from_bytes(peer_sock.recv(4), 'big')
                if not metadata_size:
                    break
                
                metadata = json.loads(peer_sock.recv(metadata_size).decode())
                filename = metadata['filename']
                filesize = metadata['filesize']
                
                # Receive file data
                filepath = os.path.join('downloads', filename)
                with open(filepath, 'wb') as f:
                    remaining = filesize
                    while remaining:
                        chunk = peer_sock.recv(min(4096, remaining))
                        if not chunk:
                            break
                        f.write(chunk)
                        remaining -= len(chunk)
                
                logging.info(f"Received file from {peer_addr}: {filepath}")
                
            except Exception as e:
                logging.error(f"Peer connection error with {peer_addr}: {e}")
                break
        
        peer_sock.close()
        self.peers.pop(peer_addr, None)

    def send_file(self, filepath):
        """Send file to all connected peers"""
        if not os.path.exists(filepath):
            logging.error(f"File not found: {filepath}")
            return

        try:
            filename = os.path.basename(filepath)
            filesize = os.path.getsize(filepath)
            
            metadata = {
                'filename': filename,
                'filesize': filesize
            }
            metadata_bytes = json.dumps(metadata).encode()
            
            # Copy peers dict to avoid modification during iteration
            peers_copy = self.peers.copy()
            for peer_addr, peer_sock in peers_copy.items():
                try:
                    # Send metadata
                    peer_sock.sendall(len(metadata_bytes).to_bytes(4, 'big'))
                    peer_sock.sendall(metadata_bytes)
                    
                    # Send file data
                    with open(filepath, 'rb') as f:
                        while chunk := f.read(4096):
                            peer_sock.sendall(chunk)
                    
                    logging.info(f"Sent {filename} to {peer_addr}")
                except Exception as e:
                    logging.error(f"Failed to send file to {peer_addr}: {e}")
                    self.peers.pop(peer_addr, None)
                    
        except Exception as e:
            logging.error(f"File send error: {e}")

    def process_file_queue(self):
        """Process queued files for sending"""
        while self.is_running:
            try:
                if not self.file_queue.empty():
                    filepath = self.file_queue.get()
                    self.send_file(filepath)
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Queue processing error: {e}")

class UIManager:
    def __init__(self):
        self.window_name = "P2P File Transfer"
        self.recent_screenshot = None
        self.recent_received = None
        self.screenshot_time = None
        self.received_time = None
        self.recent_screenshot_path = None
        self.recent_received_path = None
        
        # Define click regions
        self.screenshot_region = None
        self.received_region = None
        
    def create_info_panel(self, frame, screenshot_count, peer_count):
        """Create an info panel overlay"""
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text with better formatting
        cv2.putText(
            frame,
            f"Screenshots: {screenshot_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"Connected Peers: {peer_count}",
            (250, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Add gesture instructions
        cv2.putText(
            frame,
            "Open hand -> Close hand: Take Screenshot",
            (500, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
            
    def update_recent_screenshot(self, filepath):
        """Update the recent screenshot preview"""
        try:
            img = cv2.imread(filepath)
            if img is not None:
                self.recent_screenshot = cv2.resize(img, (320, 180))
                self.screenshot_time = datetime.now()
                self.recent_screenshot_path = filepath
        except Exception as e:
            logging.error(f"Error loading screenshot: {e}")
            
    def update_recent_received(self, filepath):
        """Update the recent received image preview"""
        try:
            img = cv2.imread(filepath)
            if img is not None:
                self.recent_received = cv2.resize(img, (320, 180))
                self.received_time = datetime.now()
                self.recent_received_path = filepath
        except Exception as e:
            logging.error(f"Error loading received image: {e}")
            
    def add_preview_panels(self, frame):
        """Add preview panels for recent screenshots and received images"""
        if frame.shape[1] < 1280 or frame.shape[0] < 720:
            return frame
            
        panel_width = 340
        right_edge = frame.shape[1]
        
        # Create right panel for previews
        cv2.rectangle(frame, (right_edge-panel_width, 0), (right_edge, frame.shape[0]), (30, 30, 30), -1)
        
        y_offset = 70
        # Recent screenshot
        cv2.putText(frame, "Recent Screenshot (click to open):", (right_edge-panel_width+10, y_offset-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if self.recent_screenshot is not None:
            preview_region = [y_offset, y_offset+180, right_edge-330, right_edge-10]
            frame[preview_region[0]:preview_region[1], preview_region[2]:preview_region[3]] = self.recent_screenshot
            self.screenshot_region = preview_region
            if self.screenshot_time:
                time_str = self.screenshot_time.strftime("%H:%M:%S")
                cv2.putText(frame, f"Time: {time_str}", (right_edge-panel_width+10, y_offset+200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
        # Recent received
        y_offset = 300
        cv2.putText(frame, "Recent Received (click to open):", (right_edge-panel_width+10, y_offset-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if self.recent_received is not None:
            preview_region = [y_offset, y_offset+180, right_edge-330, right_edge-10]
            frame[preview_region[0]:preview_region[1], preview_region[2]:preview_region[3]] = self.recent_received
            self.received_region = preview_region
            if self.received_time:
                time_str = self.received_time.strftime("%H:%M:%S")
                cv2.putText(frame, f"Time: {time_str}", (right_edge-panel_width+10, y_offset+200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    def handle_click(self, event, x, y, flags, param):
        """Handle mouse clicks on preview panels"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.screenshot_region and self.recent_screenshot_path:
                if (self.screenshot_region[2] <= x <= self.screenshot_region[3] and 
                    self.screenshot_region[0] <= y <= self.screenshot_region[1]):
                    self.open_image(self.recent_screenshot_path)
                    
            if self.received_region and self.recent_received_path:
                if (self.received_region[2] <= x <= self.received_region[3] and 
                    self.received_region[0] <= y <= self.received_region[1]):
                    self.open_image(self.recent_received_path)

    def open_image(self, filepath):
        """Open image in a new window"""
        if os.path.exists(filepath):
            img = cv2.imread(filepath)
            if img is not None:
                window_name = os.path.basename(filepath)
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, img)

def create_directories():
    """Create necessary directories"""
    for directory in ['screenshots', 'downloads']:
        os.makedirs(directory, exist_ok=True)

def is_hand_closed(hand_landmarks, mp_hands):
    """Detect closed hand gesture"""
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
    return sum(
        1 for tip, mcp in zip(finger_tips, finger_mcps)
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mcp].y
    ) >= 3

def is_hand_open(hand_landmarks, mp_hands):
    """Detect open hand gesture"""
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
    return sum(
        1 for tip, mcp in zip(finger_tips, finger_mcps)
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y
    ) >= 3

def main():
    # Initialize
    create_directories()
    p2p = P2PFileTransfer()
    p2p.start()
    ui_manager = UIManager()

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.critical("Could not open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    screenshot_count = 0
    previous_state = None
    state_change_time = time.time()

    # Monitor downloads folder for new files
    def watch_downloads():
        last_check = set()
        while True:
            current_files = set(os.listdir('downloads'))
            new_files = current_files - last_check
            if new_files:
                newest_file = max([os.path.join('downloads', f) for f in new_files], key=os.path.getctime)
                ui_manager.update_recent_received(newest_file)
            last_check = current_files
            time.sleep(1)

    threading.Thread(target=watch_downloads, daemon=True).start()

    try:
        # Create a named window and set it to fullscreen
        cv2.namedWindow(ui_manager.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(ui_manager.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Set up mouse callback
        cv2.setMouseCallback(ui_manager.window_name, ui_manager.handle_click)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Create status indicator
            status_color = (0, 255, 0) if len(p2p.peers) > 0 else (0, 0, 255)
            cv2.circle(frame, (frame.shape[1]-30, 30), 10, status_color, -1)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks with better visibility
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                    )

                    current_time = time.time()
                    if is_hand_closed(hand_landmarks, mp_hands):
                        if previous_state == 'open' and current_time - state_change_time < 1.5:
                            # Visual feedback for screenshot
                            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), 3)
                            
                            # Take screenshot
                            screenshot_count += 1
                            screenshot_path = os.path.join(
                                'screenshots',
                                f'screenshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                            )
                            with mss() as sct:
                                sct.shot(output=screenshot_path)
                            p2p.file_queue.put(screenshot_path)
                            ui_manager.update_recent_screenshot(screenshot_path)
                            logging.info(f"Screenshot captured: {screenshot_path}")
                        previous_state = 'closed'
                        state_change_time = current_time
                    elif is_hand_open(hand_landmarks, mp_hands):
                        previous_state = 'open'
                        state_change_time = current_time

            # Add UI elements
            ui_manager.create_info_panel(frame, screenshot_count, len(p2p.peers))
            ui_manager.add_preview_panels(frame)

            # Add keyboard controls info
            cv2.putText(
                frame,
                "Press 'Q' to quit | 'F' to toggle fullscreen",
                (frame.shape[1]-500, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1
            )

            cv2.imshow(ui_manager.window_name, frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                # Toggle fullscreen
                is_fullscreen = cv2.getWindowProperty(ui_manager.window_name, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(
                    ui_manager.window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL if is_fullscreen == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN
                )

    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        p2p.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()