import cv2
import auxFunc
import numpy as np
from matplotlib import pyplot as plt
import math

write_flag = True

def getFilenames(n):
	pathnames = []

	#Se componen los pathnames de todas las imágenes
	for i in range(0,n):
		pathnames.append("imagenes/" + str(i) + ".png")

	return pathnames


def BolsaPalabras(filenames, des_centroides):
	centroide_tupla = []

	#Se inicializa la bolsa de palabras a 0
	bolsa_palabras = np.zeros((len(filenames), len(des_centroides)))

	#Se crea el detector SIFT por defecto
	sift = cv2.xfeatures2d.SIFT_create()

	#Se crea el BFMatcher
	matcher = cv2.BFMatcher_create(crossCheck = 0)

	#Para cada imagen se rellena su bolsa de palabras
	for i in range(0, len(filenames)):

		#Se lee la imagen
		img = cv2.imread(filenames[i], cv2.IMREAD_GRAYSCALE)

		#Se obtienen los keyPoints y los descriptores de img
		kp, des = sift.detectAndCompute(img, None)

		#Se normaliza cada descriptor
		for d in des:
			d = np.array(d)
			d /= np.linalg.norm(d,2)
		
		#Se hace matching entre las palabras visuales y los descriptores
		matches = matcher.match(des, des_centroides)

		#Se rellena la bolsa de palabras
		for m in matches:
			bolsa_palabras[i, m.trainIdx] += 1

		print("Añadido " + filenames[i] + " a la bolsa de palabras")

	return bolsa_palabras
		
def InvIndex(bolsa):
	index = []

	#Se crea una lista de listas. Cada una de estas listas contiene las imagenes en las que aparece el descriptor correspondiente
	for j in range(0, len(bolsa[0])):
		index.append([])

	#Se rellena el indice
	for i in range(0, len(bolsa)):
		for j in range(0, len(bolsa[0])):
			if(bolsa[i,j] >= 1):
				index[j].append(i)

	return index

def similarity(des1, des2):
	array1 = np.array(des1)
	array1 /= np.linalg.norm(array1,2)
	
	array2 = np.array(des2)
	array2 /= np.linalg.norm(array2,2)

	return np.dot(array1, array2)
	

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
	

	if(write_flag):
		cv2.imwrite("Prueba.png" ,img_matches)
	
	# Pintar la imagen
	img_matches = cv2.cvtColor(img_matches, cv2.COLOR_RGB2BGR)
	plt.title("Match Region")
	plt.imshow(img_matches)
	plt.show()

def Ejercicio2(filenames, predict_filenames):
	#Umbral de similaridad
	umbral = 0.51

	#Fichero que contiene las palabras visuales
	kmeans2000 = "kmeanscenters2000.pkl"

	#Obtenemos los datos a partir del fichero
	accuracy, labels, dictionary = auxFunc.loadDictionary(kmeans2000)

	#Normalizamos los descriptores
	for word in dictionary:
		word = np.array(word)
		word /= np.linalg.norm(word,2)

	#Map filename to indice
	filename_to_idx = {}


	#Rellenamos el map
	for i in range(0, len(filenames)):
		filename_to_idx[filenames[i]] = i

	#Map descriptor de palabra visual to indice
	centroide_to_idx = {}

	#Rellenamos el map
	for i in range(0, len(dictionary)):
		centroide_to_idx[tuple(dictionary[i])] = i

	#Creamos la bolsa de palabras
	bolsa = BolsaPalabras(filenames, dictionary)

	#Creamos el indice
	index = InvIndex(bolsa)

	sift = cv2.xfeatures2d.SIFT_create()
	matcher = cv2.BFMatcher_create(crossCheck = 0)

	for label in predict_filenames:
		idx_label = filename_to_idx[label]
		bolsa_label = bolsa[idx_label]
		posible_img = []
		related_img = []

		img_label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)

		kp, des = sift.detectAndCompute(img_label, None)
		matches = matcher.match(des, dictionary)

		for m in matches:
			for img in index[m.trainIdx]:
				if(img not in posible_img and img != idx_label):
					posible_img.append(img)


		for img in posible_img:
			related_img.append((img, similarity(bolsa[img], bolsa_label)))

		related_img = sorted(related_img, key = lambda x : x[1], reverse = True)
		
		names_img = []
		for img in related_img[:5]:
			names_img.append(filenames[img[0]])

		print(label + ": " + str(names_img) + "\n")
			

def main():
	filenames = getFilenames(400)
	predict_filenames = [filenames[15], filenames[23], filenames[111]]

	print("Comienza Ejercicio 1: ")
	#Ejercicio1(filenames[0], filenames[2])

	print("Comienza Ejercicio 2: ")
	Ejercicio2(filenames, predict_filenames)


if __name__ == '__main__':
	main()
