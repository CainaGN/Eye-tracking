import cv2
import dlib
import pyautogui
import numpy as np

# Configuração inicial
pyautogui.FAILSAFE = False  # Desabilitar o fail-safe para evitar travamentos

# Carregar o detector de rosto e o modelo de landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Função para calcular o vetor da cabeça
def calculate_head_direction(landmarks):
    nose_tip = np.array([landmarks.part(30).x, landmarks.part(30).y])  # Ponta do nariz
    chin = np.array([landmarks.part(8).x, landmarks.part(8).y])  # Queixo
    left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])  # Canto do olho esquerdo
    right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])  # Canto do olho direito
    
    # Calcular o vetor médio para direção da cabeça
    horizontal = (left_eye + right_eye) / 2 - nose_tip
    vertical = chin - nose_tip
    return horizontal, vertical

# Função para mapear coordenadas do rosto para a tela
def map_to_screen(pos, calibration_data, screen_size):
    # Evitar valores inválidos na calibragem
    if calibration_data["left"][0] == calibration_data["right"][0]:
        calibration_data["right"][0] += 1
    if calibration_data["top"][1] == calibration_data["bottom"][1]:
        calibration_data["bottom"][1] += 1
    
    x = np.interp(pos[0], [calibration_data["right"][0], calibration_data["left"][0]], [10, screen_size[0] - 10])
    y = np.interp(pos[1], [calibration_data["top"][1], calibration_data["bottom"][1]], [10, screen_size[1] - 10])
    return np.clip([x, y], 0, screen_size)

# Parâmetros de calibragem
calibration_data = {"top": None, "bottom": None, "right": None, "left": None}
calibration_phase = ["top", "bottom", "left", "right"]
calibration_text = ["cima", "baixo", "direita", "esquerda"]
current_phase = 0
calibrating = True

# Iniciar captura de vídeo
cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        if calibrating:
            # Mostrar instruções para calibragem
            cv2.putText(frame, f"Olhe para {calibration_text[current_phase]} e pressione 'c'",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Se pressionar 'c', salvar coordenadas
            if cv2.waitKey(1) & 0xFF == ord('c'):
                calibration_data[calibration_phase[current_phase]] = np.array(
                    [landmarks.part(30).x, landmarks.part(30).y])
                current_phase += 1

                # Finalizar calibragem após os quatro pontos
                if current_phase >= len(calibration_phase):
                    calibrating = False
                    print("Calibragem concluída:", calibration_data)
            break

        if not calibrating:
            # Calcular direção da cabeça
            horizontal, vertical = calculate_head_direction(landmarks)
            nose_position = np.array([landmarks.part(30).x, landmarks.part(30).y])

            # Mapear a direção da cabeça para a tela
            screen_pos = map_to_screen(nose_position, calibration_data, (screen_width, screen_height))
            
            # Introduzir suavidade nos movimentos (limite de velocidade)
            current_mouse_pos = pyautogui.position()
            new_mouse_pos = [int(current_mouse_pos[0] + (screen_pos[0] - current_mouse_pos[0]) * 0.2),
                             int(current_mouse_pos[1] + (screen_pos[1] - current_mouse_pos[1]) * 0.2)]
            # criar uma variável temporária newer_mouse_pos para armazenar o valor de new_mouse_pos[0] * -1
            pyautogui.moveTo(new_mouse_pos[0], new_mouse_pos[1])

            # Exibir posição do cursor
            cv2.putText(frame, f"Cursor: {new_mouse_pos}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Desenhar os landmarks
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    # Mostrar o frame
    cv2.imshow('Head Direction Calibration', frame)

    # Sair ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar captura e fechar janelas
cap.release()
cv2.destroyAllWindows()

