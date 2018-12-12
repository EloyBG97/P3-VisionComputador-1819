import cv2
import auxFunc
import numpy as np
from matplotlib import pyplot as plt

def max_x(points):
	max = points[0][0]

	for pt in points:
		if pt[0] > max:
			max = pt[0]

	return max

def min_x(points):
	min = points[0][0]

	for pt in points:
		if(pt[0] < min):
			min = pt[0]

	return  min

def max_y(points):
        max = points[0][1]
        for pt in points:
                if pt[1] > max:
                        max = pt[1]

        return max

def min_y(points):
        min = points[0][1]

        for pt in points:
                if(pt[1] < min):
                        min = pt[1]

        return  min


def Ejercicio1(filename1, filename2):
	# Abrir imagen1
	image1 = cv2.imread(filename1)
	gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

	# Abrir imagen2
	image2 = cv2.imread(filename2)
	gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

	# Crear el detector SIFT
	sift = cv2.xfeatures2d.SIFT_create()

	#Extraer region de la imagen image1
	refPt1 = auxFunc.extractRegion(image1)

	#Obtener las coordenadas de los vertices de la region
	Mx = max_x(refPt1)
	mx = min_x(refPt1)
	My = max_y(refPt1)
	my = min_y(refPt1)

	# Marcar la region sobre la imagen 1
	image1 = cv2.rectangle(image1, (mx, my), (Mx, My), (255,0,0))

	#Construyo la mascara que define la region sobre la imagen
	mask = np.zeros(gray1.shape, np.uint8)
	mask[my:My, mx:Mx] = 1

	#Obtener los KeyPoints y Descriptores
	kp1, des1 = sift.detectAndCompute(gray1, mask)
	kp2, des2 = sift.detectAndCompute(gray2, None)

	# Creamos BFMatcher
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2, k=2)

	# Calcular inlinners y añadirlos a good
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append([m])

	# Dibujar los matches sobre la pareja de imagenes
	k = 25
	img_matches = cv2.drawMatchesKnn(image1,kp1,image2,kp2,good[:k], None, flags=2)
	

	# Pintar la imagen
	img_matches = cv2.cvtColor(img_matches, cv2.COLOR_RGB2BGR)
	plt.title("Match Region")
	plt.imshow(img_matches)
	plt.show()

	


def main():
	filename1 = "./imagenes/156.png"
	filename2 = "./imagenes/157.png"

	print("Comienza Ejercicio 1: ")
	Ejercicio1(filename1, filename2)


if __name__ == '__main__':
	main()
