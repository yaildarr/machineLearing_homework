import cv2
import mediapipe as mp
from deepface import DeepFace




owner_name = "Ildar"
owner_surname = "Kharisov"
owner_image_path = "img.png"

# Инициализация OpenCV Haar Cascade для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Инициализация MediaPipe для обнаружения рук
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Захват видео с веб-камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Функция для подсчета поднятых пальцев
def count_fingers(hand_landmarks):
    count = 0
    # Большой палец
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        count += 1
    # Остальные пальцы
    for tip_id in [8, 12, 16, 20]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            count += 1
    return count

# Функция для распознавания лица
def recognize_face(frame, face_roi):
    # Проверяем, является ли лицо владельцем
    result = DeepFace.verify(face_roi, owner_image_path, model_name="Facenet", enforce_detection=False)
    if result["verified"]:
        return "owner"
    else:
        return "unknown"


# Функция для анализа эмоции
def analyze_emotion(face_roi):
    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
    dominant_emotion = result[0]['dominant_emotion']
    return dominant_emotion


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование кадра в RGB для MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обнаружение рук
    hand_results = hands.process(rgb_frame)

    # Обнаружение лиц
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Обработка лиц
    for (x, y, w, h) in faces:
        # Выделяем лицо прямоугольником
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Извлекаем область лица для распознавания
        face_roi = frame[y:y + h, x:x + w]

        # Распознавание лица
        identity = recognize_face(frame, face_roi)

        # Обработка рук и пальцев
        finger_count = 0
        if hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_count = count_fingers(hand_landmarks)

        # текст для отображения
        text = ""
        if identity == "owner":
            if finger_count == 1:
                text = f"Firstname: {owner_name}"
            elif finger_count == 2:
                text = f"Lastname: {owner_surname}"
            elif finger_count == 3:
                emotion = analyze_emotion(face_roi)
                text = f"Emotion: {emotion}"
            else:
                text = "Owner"
        else:
            text = "Unknown"

        # текст под лицом
        cv2.putText(frame, text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face and Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
hands.close()
cv2.destroyAllWindows()
