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
		# Obtener todas las marcas de referencia en la primera mano visible
		landmarks = hand_landmarks[handNo].landmark

		# Contar dedos
		fingers = []

		for lm_index in tipIds:
			# Obtener los valores de la psosición "y" de la punta y parte inferior del dedo
			finger_tip_y = landmarks[lm_index].y 
			finger_bottom_y = landmarks[lm_index - 2].y

			# Verificar si algun dedo está abierto o cerrado
			if lm_index !=4:
				if finger_tip_y < finger_bottom_y:
					fingers.append(1)


				if finger_tip_y > finger_bottom_y:
					fingers.append(0)

		totalFingers = fingers.count(1)

		# Pellizco

		# Dibujar una línea entre la punta del dedo y la punta del pulgar
		finger_tip_x = int((landmarks[8].x)*width)
		finger_tip_y = int((landmarks[8].y)*height)

		thumb_tip_x = int((landmarks[4].x)*width)
		thumb_tip_y = int((landmarks[4].y)*height)

		cv2.line(image, (finger_tip_x, finger_tip_y),(thumb_tip_x, thumb_tip_y),(255,0,0),2)

		# Dibujar un círculo en el centro de la línea entre la punta del dedo y la punta del pulgar
		center_x = int((finger_tip_x + thumb_tip_x )/2)
		center_y = int((finger_tip_y + thumb_tip_y )/2)

		cv2.circle(image, (center_x, center_y), 2, (0,0,255), 2)

		# Calcular la distancia entre la punta del dedo y la punta del pulgar
		distance = math.sqrt(((finger_tip_x - thumb_tip_x)**2) + ((finger_tip_y - thumb_tip_y)**2))

		# print("Distancia: ", distance)
		
		print("Tamaño de la pantalla de la computadora: ",screen_width, screen_height, " Tamaño de la ventana del output: ", width, height)
		print("Posición del mouse: ", mouse.position, "Posición del centro de la línea de las puntas: ", center_x, center_y)

		# Establecer la posición del mouse en la pantalla relativa al tamaño de la ventana del output
		relative_mouse_x = (center_x/width)*screen_width
		relative_mouse_y = (center_y/height)*screen_height
		
		mouse.position = (relative_mouse_x, relative_mouse_y)

		# Verificar las condiciones de la formación del pellizco
		if distance > 40:
			if pinch == True:
				pinch = False			
				mouse.release(Button.left)

		if distance <= 40 :
			if(pinch==False):
				pinch=True
				mouse.press(Button.left)


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
