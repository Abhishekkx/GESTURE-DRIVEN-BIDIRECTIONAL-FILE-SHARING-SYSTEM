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

def create_directories():
    """Create necessary directories"""
    for directory in ['screenshots', 'downloads']:
        os.makedirs(directory, exist_ok=True)

def main():
    # Initialize
    create_directories()
    p2p = P2PFileTransfer()
    p2p.start()

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

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    current_time = time.time()
                    if is_hand_closed(hand_landmarks, mp_hands):
                        if previous_state == 'open' and current_time - state_change_time < 1.5:
                            # Take screenshot
                            screenshot_count += 1
                            screenshot_path = os.path.join(
                                'screenshots',
                                f'screenshot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
                            )
                            with mss() as sct:
                                sct.shot(output=screenshot_path)
                            p2p.file_queue.put(screenshot_path)
                            logging.info(f"Screenshot captured: {screenshot_path}")
                        previous_state = 'closed'
                        state_change_time = current_time
                    elif is_hand_open(hand_landmarks, mp_hands):
                        previous_state = 'open'
                        state_change_time = current_time

            cv2.putText(
                frame,
                f"Screenshots: {screenshot_count} | Peers: {len(p2p.peers)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.imshow("P2P File Transfer", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        p2p.stop()
        cap.release()
        cv2.destroyAllWindows()

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

if __name__ == "__main__":
    main()