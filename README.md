# Projekt_Wykryw_Twarzy
from fer.fer import FER
import cv2
import matplotlib.pyplot as plt

# Inicjalizacja modelu
detector = FER(mtcnn=True)

# Wczytanie obrazu
image_path = "neutral_dziad.png"
img = cv2.imread(image_path)

if img is None:
    raise ValueError("Nie można wczytać obrazu. Sprawdź nazwę pliku.")

# Predykcja emocji
result = detector.detect_emotions(img)

if not result:
    print("Nie wykryto twarzy")
else:
    data = result[0]

    box = data["box"]
    emotions = data["emotions"]

    # Dominująca emocja
    dominant_emotion = max(emotions, key=emotions.get)

    print("\nDominująca emocja:", dominant_emotion)

    # Wszystkie emocje
    print("\nWszystkie emocje:")
    for k, v in emotions.items():
        print(f"{k}: {v:.3f}")

    # TOP 3 emocje
    top3 = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]

    print("\nTop 3 emocje:")
    for emo, score in top3:
        print(f"{emo}: {score:.3f}")

    # Interpretacja
    print("\nInterpretacja:")

    if dominant_emotion == "happy":
        print("Użytkownik wygląda na zadowolonego")
    elif dominant_emotion == "sad":
        print("Wykryto smutek")
    elif dominant_emotion == "angry":
        print("Wykryto złość")
    elif dominant_emotion == "neutral":
        print("Stan neutralny")
    else:
        print("Inna emocja:", dominant_emotion)

    # Rysowanie boxa
    x, y, w, h = box

    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        img,
        dominant_emotion,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Wyświetlenie
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Emotion Recognition Result")
    plt.show()
