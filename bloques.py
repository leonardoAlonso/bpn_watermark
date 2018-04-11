import cv2
import numpy as np
import backNN as bpn
from scipy.fftpack import dct

def separa_bloque(gray):
    matrices = []
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            bloque = gray[i*8:(i+1)*8,j*8:(j+1)*8]
            if(bloque.shape[0] != 0 and bloque.shape[1]):
                matrices.append(bloque)
    return matrices

def return_image(vector):
    m = 0
    matriz = np.zeros((256,256), np.uint8)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            bloque = matriz[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            if (bloque.shape[0] != 0 and bloque.shape[1]):
                matriz[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = vector[m]
                m += 1
                if m > 1024:
                    break
    matriz = matriz.astype(np.uint8)
    return matriz

def return_marca(vector):
    m = 0
    matriz = np.zeros((32, 32), np.uint8)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            matriz[i,j] = vector[m]
            m += 1
            if m > 1024:
                break
    matriz = matriz.astype(np.uint8)
    return matriz

def dct_b(bloque):
    quantization = 16.0
    coef = dct(bloque, type = 2, n = None, axis=-1, norm='ortho',overwrite_x=False)
    return round(coef[0][0]/quantization)

def average(bloque):
    avg = 0
    for i in(0, bloque.shape[0] - 1):
        for j in (0, bloque.shape[1] - 1):
            avg = avg + bloque[i][j]
    return avg/bloque.shape[0]*bloque.shape[1]

def return_output_nn(pat):
    NN = bpn.NN(1, 5, 1)
    one = []
    one.append(pat)
    #print(one)
    inputs = one[0]
    NN.train(one)
    salida = NN.runNN(inputs[0])
    salida = salida[0]
    return round(salida,2), NN

def suma_bloque(bloques):
    suma_v = []
    for i in bloques:
        suma = 0
        for x in range(i.shape[0]):
            for y in range(i.shape[1]):
                suma += i[x][y]
        suma_v.append(suma)
    return suma_v

def lista_marca(marca):
    lista = []
    for i in range(marca.shape[0]):
        for j in range(marca.shape[1]):
            lista.append(marca[i][j])
    return lista

def metricPSNR (image1,image2):
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    dif = np.sum((image1 - image2) ** 2)
    mse = dif / (image1.shape[0] * image2.shape[1] * image2.shape[2])
    psnr = 10 * np.log10((255*255)/mse)
    return psnr

def bcr(marca_or, marca_ex):
    sum_or = 0
    sum_ex = 0
    for i in range(marca_or.shape[0]):
        for j in range(marca_or.shape[1]):
            sum_or += marca_or[i][j]
            sum_ex += marca_ex[i][j]
    be = 1-((sum_or^sum_ex)/(marca_or.shape[0]*marca_or.shape[1]))
    return be*100

def do_binary(image):
    binary = np.zeros(image.shape, np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] == 255:
                binary[i,j] = 1
            else:
                image[i,j] = 0
    return binary

def main():
    '''Imagen original'''
    imagen = cv2.imread('Lena.tiff')
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256,256), interpolation = cv2.INTER_CUBIC)
    gray = gray.astype(np.float64)
    '''Marca de agua para incertar'''
    marca = cv2.imread('marca.png')
    marca_gray = cv2.cvtColor(marca, cv2.COLOR_BGR2GRAY)
    marca_gray = cv2.resize(marca_gray, (32,32), interpolation = cv2.INTER_CUBIC)
    ret,binary = cv2.threshold(marca_gray,127,255,cv2.THRESH_BINARY)
    binary = binary.astype(np.float64)


    '''Obtencion de los coeficientes de dc y los promedios
    para el entrenamiento de la red neuronal'''
    coef = []
    avg = []
    bloques = separa_bloque(gray)

    i = 0
    while i < len(bloques):
        coef.append(dct_b(bloques[i]))
        i = i + 1
    #normalizar factores de dct
    max_dct = max(coef)
    i = 0
    while i < len(coef):
        coef[i] = coef[i]/max_dct
        i = i + 1

    i = 0
    while i < len(bloques):
        avg.append(average(bloques[i]))
        i = i + 1
    #normalizar datos del promedio
    max_avg = max(avg)
    i = 0
    while i < len(bloques):
        avg[i] = avg[i]/max_avg
        i = i + 1

    #entrenamiento de red neuronal
    pat = [i for i in range(0,len(coef))]
    for i in range(0, len(coef)):
        pat[i] = [[coef[i]],[avg[i]]]

    salidas = []
    net = []
    for i in range(0, len(pat)):
        s, network = return_output_nn(pat[i])
        salidas.append(s)
        net.append(network)
    #multiplicar promedio y salidas obtenidas por el valor maximo del promedio
    i = 0
    while i < len(bloques):
        avg[i] = avg[i]*max_avg
        i = i + 1
    i = 0
    while i < len(bloques):
        salidas[i] = salidas[i]*(max_dct*16)
        i = i + 1
    #incrustacion de la marca de agua
    print("Incrustando la marca...")
    suma_v = suma_bloque(bloques)

    img_marcada = np.zeros((gray.shape), np.uint8)
    bloques_marcados = separa_bloque(img_marcada)
    lista_binaria = lista_marca(binary)
    lista_bloques_marcador = []
    q = 16.0
    for i in range(0,len(bloques_marcados)):
        bloque_m = bloques_marcados[i]
        bloque_o = bloques[i]
        if lista_binaria[i] == 255:
            for x in range(bloque_m.shape[0]):
                for y in range(bloque_m.shape[1]):
                    bloque_m[x,y] = bloque_o[x,y] + (round(((0.25*8*q + (64 * salidas[i]) - (suma_v[i])))/suma_v[i]))
                    #print("valor 1",round((0.25*8*q + 64 * salidas[i] - suma_v[i])/suma_v[i]))
        else:
            for x in range(bloque_m.shape[0]):
                for y in range(bloque_m.shape[1]):
                    bloque_m[x,y] = bloque_o[x,y] - (round(((0.25*8*q + (64 * salidas[i]) - (suma_v[i])))/suma_v[i]))
                    #print("valor 0", -round((0.25 * 8 * q + 64 * salidas[i] - suma_v[i]) / suma_v[i]))
        lista_bloques_marcador.append(bloque_m)

    img_marcada = return_image(lista_bloques_marcador)
    print("Terminado")

    #extraccion de la marca
    print("Extrayendo la marca...")
    bloques_img_marcada = separa_bloque(img_marcada)
    coef = []
    avg_n = []
    salidas_n = []
    i = 0
    while i < len(bloques_img_marcada):
        coef.append(dct_b(bloques_img_marcada[i]))
        i = i + 1
    max_dct = max(coef)
    i = 0
    while i < len(coef):
        coef[i] = coef[i]/max_dct
        i = i + 1
    for i in range(len(net)):
        c = []
        c.append(coef[i])
        salidas_n.append(net[i].runNN(c))
    print(max_avg)
    sal = []
    for i in salidas_n:
        sal.append(float(round(i[0]*(max_avg))))

    print(sal)
    i = 0
    while i < len(bloques_img_marcada):
        avg_n.append(average(bloques_img_marcada[i]))
        i = i + 1
    print(avg_n)
    marca_extraida = []
    for i in range(len(avg_n)):
        if avg_n[i] >= sal[i]:
            marca_extraida.append(255)
        else:
            marca_extraida.append(0)
    marca_extraida = return_marca(marca_extraida)
    print("Terminado")
    print("PSNR de gris y imagen marcada",metricPSNR(gray, img_marcada))
    #print("PSNR de marca y marca extraida",metricPSNR(marca_extraida, binary))
    cv2.imshow("Imagen marcada", img_marcada)
    cv2.imshow("marca extraida", marca_extraida)
    gray = gray.astype(np.uint8)
    cv2.imshow("original", gray)
    cv2.imshow("marca", binary)
    cv2.imshow("diferencia", gray-img_marcada)
    cv2.imshow("diferencia marca", binary-marca_extraida)

    print(lista_binaria)
    print(do_binary(marca_extraida))
    cv2.waitKey(0)
    cv2.destroyAllWindows()