import cv2
import numpy as np

EMOTIONS = ["Happy", "Sad", "Angry", "Neutral", "Surprise", "Fear", "Disgust"]

SUGGESTIONS = {
    "Happy": "Great mood! Keep that positive energy going.",
    "Sad": "Take a short break, breathe, and do something you enjoy.",
    "Angry": "Pause for a moment and take a few slow breaths.",
    "Neutral": "You look calm. This could be a good time to focus.",
    "Surprise": "Something caught your attention!",
    "Fear": "Take a breath and give yourself a moment to reset.",
    "Disgust": "Something seems unpleasant. Take a comfortable break.",
}


def predict_mood(face):
    """Educational demo classifier based on simple image features."""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    if brightness > 165 and contrast > 55:
        return "Happy"
    if brightness < 75:
        return "Sad"
    if contrast > 65:
        return "Surprise"
    if brightness < 105:
        return "Angry"
    return "Neutral"


def main():
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Unable to access the camera.")
        return

    mood = "Waiting..."
    history = []

    print("AI Emotion & Mood Assistant")
    print("Press Q to exit.")

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )

        for x, y, w, h in faces:
            face = frame[y:y + h, x:x + w]
            mood = predict_mood(face)

            if not history or history[-1] != mood:
                history.append(mood)

            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (230, 180, 70), 2
            )
            cv2.putText(
                frame, mood, (x, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 180, 70), 2
            )

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 92), (28, 25, 22), -1)
        cv2.putText(
            frame, "EMOTION & MOOD ASSISTANT", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2
        )
        cv2.putText(
            frame, f"Mood: {mood}", (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 180, 70), 2
        )

        cv2.imshow("Emotion & Mood Assistant", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("\nSession Summary")
    print("----------------")
    print("Moods:", ", ".join(history) if history else "None")
    print("Last mood:", mood)
    print("Suggestion:", SUGGESTIONS.get(mood, "Take a moment and reset."))


if __name__ == "__main__":
    main()
