
Gesture-Driven Bidirectional File Sharing System
📌 Overview
The Gesture-Driven Bidirectional File Sharing System is an innovative touchless file transfer solution that leverages hand gesture recognition for seamless communication between devices. The system utilizes MediaPipe for real-time hand tracking and socket programming for efficient, bidirectional file transfers.

This project redefines digital interaction by eliminating the need for keyboards, mice, or touchscreens, allowing users to perform essential file management tasks like capturing screenshots and transferring files using simple hand gestures.

✨ Features
✔ Touchless Interaction – Transfer files using hand gestures without any physical input.
✔ Real-Time Gesture Recognition – Uses MediaPipe to detect and track hand movements accurately.
✔ Bidirectional File Transfer – Enables seamless sending and receiving of files.
✔ Minimal Latency – Utilizes socket programming (TCP/IP) for fast and reliable communication.
✔ Accessibility & Inclusivity – Designed to assist users with limited mobility or those in sterile environments.
✔ Multi-Threading Support – Ensures parallel processing for gesture recognition and file transfers.

🎯 Use Cases
Office & Productivity – Touchless file sharing during presentations or remote work.
Accessibility – Helps users with disabilities interact with computers more naturally.
Gaming & Virtual Reality – Hands-free interaction for immersive experiences.
Healthcare – Enables touchless data transfer in sterile environments.
🔧 Technologies Used
Python – Core language for implementation.
MediaPipe – Hand tracking and gesture recognition.
OpenCV – Video capture and image preprocessing.
Socket Programming (TCP/IP) – Enables bidirectional file transfer.
Threading – Handles gesture detection and file transfer simultaneously.
Logging – Monitors and debugs system performance.
🏗 System Architecture
The project follows a client-server model, ensuring smooth gesture recognition and file transmission.

📌 Client-Side Components
Gesture Recognition Module – Uses MediaPipe to track and analyze hand gestures.
Command Execution Layer – Maps detected gestures to file operations.
File Transfer Module – Sends and receives files via socket programming.
📌 Server-Side Components
Connection Management – Handles incoming client connections and error handling.
File Handling Module – Manages incoming and outgoing files.
Logging Framework – Monitors file transfers and gesture recognition processes.
🖐 Supported Gestures
Gesture	Action
Open Hand → Closed Hand	Capture Screenshot & Initiate File Transfer
Closed Hand → Open Hand	Receive File
🚀 How It Works
1️⃣ Capture real-time video frames using OpenCV.
2️⃣ Detect and analyze hand landmarks using MediaPipe.
3️⃣ Recognize predefined gestures to initiate file operations.
4️⃣ Establish a TCP connection for file transfer.
5️⃣ Send and receive files in optimized chunks (4096 bytes) for efficiency.

🛠 Installation & Setup
Prerequisites
Ensure you have Python 3.7+ installed along with the required dependencies.

bash
Copy
Edit
pip install opencv-python mediapipe
Running the Project
1️⃣ Start the Server
Run the server script to listen for client connections.

bash
Copy
Edit
python server.py
2️⃣ Start the Client
Run the client script and initiate file transfer through hand gestures.

bash
Copy
Edit
python client.py
📊 Performance Metrics
Gesture Recognition Accuracy: 92.5%
File Transfer Speed: 2.4 MB/sec
Success Rate: 96.7%
CPU Usage: 15-25%
Memory Consumption: 128-256MB
🔍 Limitations & Future Improvements
✔ Expanding Gesture Vocabulary – Adding more gestures for complex file operations.
✔ Enhancing Security – Implementing encryption for secure file transfers.
✔ Multi-User Support – Allowing multiple clients to interact with the system simultaneously.
✔ Optimizing for Different Environments – Improving accuracy in low-light conditions.
