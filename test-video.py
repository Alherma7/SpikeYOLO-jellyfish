import cv2
from ultralytics import YOLO

# Cargar tu modelo entrenado
model = YOLO("/home/alher/SpikeYOLO/runs/detect/train6/weights/best.pt")

# Lista de clases (solo 1)
class_list = ["none","jellyfish"]

# Color para dibujar
color = (0, 255, 0)

# Abrir vídeo
cap = cv2.VideoCapture("Jellyfish_101_ Nat Geo Wild.mp4")

if not cap.isOpened():
    print("No se pudo abrir el video")
    exit()

cv2.namedWindow("SpikeYOLO - detección en vivo", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferencia en tiempo real
    results = model(frame, device=0)

    # Dibujar detecciones
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0]

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{class_list[cls]} {conf:.2f}",
                    (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Mostrar ventana
    cv2.imshow("SpikeYOLO - detección en vivo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

