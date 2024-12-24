import argparse
import signal
import cv2

def signal_handler(signum, frame):
    print("\nCtrl+C pressed. Releasing camera and exiting...")
    cap.release()
    cv2.destroyAllWindows()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

parser = argparse.ArgumentParser(description='Capture video from a specific camera device.')
parser.add_argument('device_number', type=int, help='The camera device number (e.g., 0 for /dev/video0)')

args = parser.parse_args()
device_path = f'/dev/video{args.device_number}'

#To see all available video devices available to cv2 from the terminal, enter this command (without the quotes):
#"v4l2-ctl --list-devices"

#To see the available capture resolutions, enter this command (without the quotes):
#"v4l2-ctl --list-formats-ext"

#You can safely quit either by pressing ctrl-c with the focus on the terminal or by pressing 'q' in the lmshow video window

# Create a VideoCapture object
#cap = cv2.VideoCapture('/dev/video0')
cap = cv2.VideoCapture(device_path)

if not cap.isOpened():
    print(f"Error: Could not open camera at {device_path}")
    exit()

# Set the resolution to 256x384
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)

# Verify the settings
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution set to: {width}x{height}")

# Capture and display video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close windows
cap.release()
cv2.destroyAllWindows()
