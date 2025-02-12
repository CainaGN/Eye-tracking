# Head Tracking e Controle de Mouse por Movimentos Faciais

Este projeto utiliza a biblioteca MediaPipe para rastrear movimentos faciais e controlar o cursor do mouse com base na direção da cabeça. Além disso, ele detecta piscadas para simular cliques do mouse. O código é escrito em Python e depende de algumas bibliotecas externas.

## Autores:

- Cainã Gonçalves Nascimento
- Glaicon Farias Felipe

## Requisitos

- Python
- Webcam funcional
- Bibliotecas Python:
  - OpenCV (`cv2`)
  - MediaPipe (`mediapipe`)
  - PyAutoGUI (`pyautogui`)
  - NumPy (`numpy`)

## Instalação

1. **Instale o Python:**  
   Baixe e instale o Python a partir do site oficial: [python.org](https://www.python.org/).

2. **Crie um ambiente virtual** (opcional, mas recomendado):

   ```bash
   python -m venv venv
   source venv/bin/activate  # No Linux/Mac
   venv\Scripts\activate     # No Windows
   ```

3. **Instale as dependências:**

   ```bash
   pip install opencv-python mediapipe pyautogui numpy
   ```

4. **Baixe o código:**  
   Salve o código Python em um arquivo, por exemplo, `head_tracking.py`.

## Como Usar

1. **Execute o código:**

   ```bash
   python head_tracking.py
   ```

2. **Calibração:**
   O programa iniciará a calibração. Siga as instruções na tela:

   - Olhe para cima e pressione a tecla `c`.
   - Olhe para baixo e pressione a tecla `c`.
   - Olhe para a esquerda e pressione a tecla `c`.
   - Olhe para a direita e pressione a tecla `c`.

   Após a calibração, o programa começará a rastrear seus movimentos faciais.

3. **Controle do Mouse:**

   - Mova a cabeça para controlar o cursor do mouse.
   - Piscar os olhos simula um clique do mouse.

