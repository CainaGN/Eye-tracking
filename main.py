import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Configuração inicial
pyautogui.FAILSAFE = False  # Desabilitar o fail-safe para evitar travamentos

# Configurações do MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Objeto para desenhar annotations
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Função para calcular o EAR (Eye Aspect Ratio)
def calculate_ear(landmarks, eye_indices):
    # Obter os pontos do olho
    eye = np.array([(landmarks[idx].x, landmarks[idx].y) for idx in eye_indices])
    # Converter para coordenadas de pixel
    eye = eye * np.array([img_width, img_height])
    # Cálculo do EAR
    A = np.linalg.norm(eye[1] - eye[5])  # Distância vertical
    B = np.linalg.norm(eye[2] - eye[4])  # Outra distância vertical
    C = np.linalg.norm(eye[0] - eye[3])  # Distância horizontal
    ear = (A + B) / (2.0 * C)
    return ear

# Função para mapear coordenadas do rosto para a tela
def map_to_screen(pos, calibration_data, screen_size):
    # Evitar valores inválidos na calibragem
    if calibration_data["left"][0] == calibration_data["right"][0]:
        calibration_data["right"][0] += 1
    if calibration_data["top"][1] == calibration_data["bottom"][1]:
        calibration_data["bottom"][1] += 1

    x = np.interp(pos[0], [calibration_data["right"][0], calibration_data["left"][0]], [10, screen_size[0] - 10])
    y = np.interp(pos[1], [calibration_data["top"][1], calibration_data["bottom"][1]], [10, screen_size[1] - 10])
    return np.clip([x, y], [0, 0], screen_size)

# Índices dos landmarks para os olhos no MediaPipe
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Ajustados para cálculo do EAR
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Ajustados para cálculo do EAR

# Parâmetros de calibragem
calibration_data = {"top": None, "bottom": None, "right": None, "left": None}
calibration_phase = ["top", "bottom", "left", "right"]
calibration_text = ["cima", "baixo", "direita", "esquerda"]
current_phase = 0
calibrating = True

# Constantes para a detecção de piscadas
EYE_AR_THRESHOLD = 0.21  # Limite para considerar uma piscada (ajuste baseado em testes)
EYE_AR_CONSEC_FRAMES = 3  # Número mínimo de frames consecutivos para detectar piscada
blink_counter = 0
blink_detected = False

# Iniciar captura de vídeo
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = frame.shape
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            if calibrating:
                # Mostrar instruções para calibragem
                cv2.putText(
                    frame,
                    f"Olhe para {calibration_text[current_phase]} e pressione 'c'",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Se pressionar 'c', salvar coordenadas
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    calibration_data[calibration_phase[current_phase]] = np.array(
                        [landmarks[1].x * img_width, landmarks[1].y * img_height]
                    )
                    current_phase += 1

                    # Finalizar calibragem após os quatro pontos
                    if current_phase >= len(calibration_phase):
                        calibrating = False
                        print("Calibragem concluída:", calibration_data)
                # Importante: precisamos de um 'break' aqui para evitar múltiplas leituras durante a calibragem
                break

            else:
                # Calcular direção da cabeça
                nose_tip = np.array([landmarks[1].x * img_width, landmarks[1].y * img_height])
                nose_position = nose_tip

                # Mapear a direção da cabeça para a tela
                screen_pos = map_to_screen(nose_position, calibration_data, (screen_width, screen_height))
                
                # Introduzir suavidade nos movimentos (limite de velocidade)
                current_mouse_pos = pyautogui.position()
                new_mouse_pos = [
                    int(current_mouse_pos[0] + (screen_pos[0] - current_mouse_pos[0]) * 0.2),
                    int(current_mouse_pos[1] + (screen_pos[1] - current_mouse_pos[1]) * 0.2)
                ]
                pyautogui.moveTo(new_mouse_pos[0], new_mouse_pos[1])

                # Exibir posição do cursor
                cv2.putText(
                    frame,
                    f"Cursor: {new_mouse_pos}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                # Detectar piscadas
                left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
                right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < EYE_AR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= EYE_AR_CONSEC_FRAMES:
                        blink_detected = True
                    blink_counter = 0

                if blink_detected:
                    pyautogui.click()  # Simular clique do mouse
                    blink_detected = False

            # Desenhar os landmarks
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

    # Mostrar o frame
    cv2.imshow('Head Direction Calibration', frame)

    # Sair ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar captura e fechar janelas
cap.release()
cv2.destroyAllWindows()
