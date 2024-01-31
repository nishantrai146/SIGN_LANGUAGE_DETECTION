import cv2
import numpy as np
import os
import mediapipe as mp

# Constants
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['A', 'B', 'C'])
no_sequences = 30
sequence_length = 30

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def load_model():
    try:
        # Load the hand detection model
        return mp_hands.Hands()
    except Exception as e:
        print(f"Error loading the hand detection model: {e}")
        return None

def mediapipe_detection(image, model):
    """
    Process an image using the MediaPipe hand detection model.

    Parameters:
    - image: The input image.
    - model: The MediaPipe hand detection model.

    Returns:
    - An image with landmarks drawn.
    - Results from the hand detection model.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    """
    Draw styled landmarks on the image.

    Parameters:
    - image: The input image.
    - results: Results from the hand detection model.
    """
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

def extract_keypoints(results):
    """
    Extract keypoints from hand landmarks.

    Parameters:
    - results: Results from the hand detection model.

    Returns:
    - Flattened array of hand landmarks.
    """
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21 * 3)
            keypoints.append(rh)
    return keypoints

def main():
    # Initialize MediaPipe Hands model
    hand_model = load_model()

    if hand_model is None:
        return

    # Open a video capture
    cap = cv2.VideoCapture(0)

    for sequence in range(no_sequences):
        sequence_data = []
        for frame_num in range(sequence_length):
            _, frame = cap.read()

            # Process the frame using MediaPipe
            frame, results = mediapipe_detection(frame, hand_model)

            # Draw landmarks on the frame
            draw_styled_landmarks(frame, results)

            # Display the frame
            cv2.imshow("Hand Gesture Recognition", frame)

            # Extract keypoints
            keypoints = extract_keypoints(results)
            sequence_data.append(keypoints)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Save captured data for the sequence (adjust as needed)
        save_data(sequence_data, sequence)

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

def save_data(sequence_data, sequence):
    # Define the directory to save data
    save_dir = os.path.join(DATA_PATH, f"Sequence_{sequence}")

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save keypoints data for each frame in the sequence
    for frame_num, keypoints in enumerate(sequence_data):
        np.save(os.path.join(save_dir, f"frame_{frame_num}.npy"), keypoints)

if __name__ == "__main__":
    main()
