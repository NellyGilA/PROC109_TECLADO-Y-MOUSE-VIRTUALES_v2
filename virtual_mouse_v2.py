import cv2
import math
import mediapipe as mp
from pynput.mouse import Button, Controller
import pyautogui

mouse=Controller()

cap = cv2.VideoCapture(0)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

(screen_width, screen_height) = pyautogui.size()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

tipIds = [4, 8, 12, 16, 20]

pinch=False

# Definir una función para contar dedos
def countFingers(image, hand_landmarks, handNo=0):
    global pinch

    if hand_landmarks:
        landmarks = hand_landmarks[handNo].landmark
        fingers = []
        
        for lm_index in tipIds:
            finger_tip_y = landmarks[lm_index].y
            finger_bottom_y = landmarks[lm_index - 2].y

            if lm_index != 4:
                if finger_tip_y < finger_bottom_y:
                    fingers.append(1)
                else:
                    fingers.append(0)
        
        totalFingers = fingers.count(1)

        # === Coordenadas de dedos ===
        finger_tip_x = int((landmarks[8].x) * width)
        finger_tip_y = int((landmarks[8].y) * height)

        thumb_tip_x = int((landmarks[4].x) * width)
        thumb_tip_y = int((landmarks[4].y) * height)

        cv2.line(image, (finger_tip_x, finger_tip_y), (thumb_tip_x, thumb_tip_y), (255, 0, 0), 2)

        center_x = int((finger_tip_x + thumb_tip_x) / 2)
        center_y = int((finger_tip_y + thumb_tip_y) / 2)
        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), 2)

        distance = math.sqrt(((finger_tip_x - thumb_tip_x)**2) + ((finger_tip_y - thumb_tip_y)**2))

        # === Movimiento del mouse ===
        relative_mouse_x = (center_x / width) * screen_width
        relative_mouse_y = (center_y / height) * screen_height  
        mouse.position = (relative_mouse_x, relative_mouse_y)

        # === Click con pellizco ===
        if distance <= 40:  # dedos juntos
            if pinch == False:
                pinch = True
                mouse.press(Button.left)
        else:
            if pinch == True:
                pinch = False
                mouse.release(Button.left)

        # === NUEVA SECCIÓN: Scroll del mouse ===
        # Si solo el índice está arriba → scroll arriba
        if fingers[0] == 1 and fingers[1] == 0:
            pyautogui.scroll(10)
            cv2.putText(image, "Scroll Up", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Si índice y medio están arriba → scroll abajo
        elif fingers[0] == 1 and fingers[1] == 1:
            pyautogui.scroll(-10)
            cv2.putText(image, "Scroll Down", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Mostrar número de dedos detectados
        text = f'Dedos: {totalFingers}'
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


# Definir una función para
def drawHandLanmarks(image, hand_landmarks):

    # Dibujar conexiones entre las marcas de referencia
    if hand_landmarks:

      for landmarks in hand_landmarks:
               
        mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)



while True:
	success, image = cap.read()
	
	image = cv2.flip(image, 1)

	# Detectar las marcas de referencia de las manos
	results = hands.process(image)

	# Obtener las marcas de referencia del resultado procesado
	hand_landmarks = results.multi_hand_landmarks

	# Dibujar las marcas de referencia
	drawHandLanmarks(image, hand_landmarks)

	# Obtener la posoción de los dedos de las manos
	countFingers(image, hand_landmarks)

	cv2.imshow("Controlador de medios", image)

	# Cerrar la ventana al presionar la barra espaciadora
	key = cv2.waitKey(1)
	if key == 27:
		break

cv2.destroyAllWindows()
