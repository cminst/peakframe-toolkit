import cv2
import sys

if len(sys.argv) < 2:
    print("Usage: python your_script_name.py <video_path>")
    sys.exit(1)

video_path = sys.argv[1]

vidcap = cv2.VideoCapture(video_path)
if not vidcap.isOpened():
    print(f"Error: Could not open video file '{video_path}'")
    sys.exit(1)

success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
