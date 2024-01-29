import os
import cv2

cap = cv2.VideoCapture(0)
directory = 'Image/'

# Create directories if they don't exist
for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    os.makedirs(os.path.join(directory, letter), exist_ok=True)

current_letter = 'A'
image_count = 0

while True:
    _, frame = cap.read()

    count = {letter: len(os.listdir(os.path.join(directory, letter))) for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}

    row = frame.shape[1]
    col = frame.shape[0]
    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
    cv2.imshow("data", frame)
    cv2.imshow("ROI", frame[40:400, 0:300])
    frame = frame[40:400, 0:300]

    interrupt = cv2.waitKey(10)

    # Check if the pressed key is 'c'
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(os.path.join(directory, current_letter, f"{count[current_letter]}.png"), frame)
        count[current_letter] += 1
        image_count += 1
        print(f'Captured image {image_count} for {current_letter}')

    # Check if the pressed key is 'q'
    elif interrupt & 0xFF == ord('q'):
        # Move to the next letter
        current_letter = chr(ord(current_letter) + 1) if current_letter != 'Z' else 'A'
        image_count = 0
        print(f'Switching to directory {current_letter}')

    # Break the loop and release the camera if 'Esc' is pressed
    elif interrupt & 0xFF == 27:  # 27 is the ASCII value for 'Esc'
        break

cap.release()
cv2.destroyAllWindows()
