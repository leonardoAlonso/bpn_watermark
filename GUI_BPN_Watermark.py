import sys
import cv2
import numpy as np
import bloques as block

from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QTextEdit, \
    QMessageBox
from PyQt5.QtCore import Qt


class GuiWatermark(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.marca = None
        self.label = QLabel()
        self.label_marca  =QLabel()
        self.net = []
        self.logs = QTextEdit()
        self.max_avg = None
        self.initUI()

    def initUI(self):
        self.label.setText('Imagen Host')
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet('border: gray; border-style:solid; border-width: 1px;')
        self.label_marca.setText("Marca")
        self.label_marca.setAlignment(Qt.AlignCenter)
        self.label_marca.setStyleSheet('border: gray; border-style:solid; border-width: 1px;')

        btn_open = QPushButton('Abir Imagen Host...')
        btn_open.clicked.connect(self.abrir_imagen)

        btn_open_marca = QPushButton('Abir Marca...')
        btn_open_marca.clicked.connect(self.abrir_marca)

        btn_insertar = QPushButton('Insertar marca')
        btn_insertar.clicked.connect(self.insertar_marca)
        btn_extraer = QPushButton('Extraer Marca')
        btn_extraer.clicked.connect(self.extraer_marca)


        top_bar = QHBoxLayout()
        top_bar.addWidget(btn_open)
        top_bar.addWidget(btn_open_marca)
        top_bar.addWidget(btn_insertar)
        top_bar.addWidget(btn_extraer)

        root = QVBoxLayout(self)
        root.addLayout(top_bar)
        root.addWidget(self.label)
        root.addWidget(self.label_marca)
        root.addWidget(self.logs)


        self.resize(500, 500)
        self.setWindowTitle('BPN_Watermark')
        self.setWindowIcon(QIcon("marca.png"))

    def abrir_imagen(self):
        self.image = None
        filename, _ = QFileDialog.getOpenFileName(None, 'Buscar Imagen', '.', 'Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)')
        if filename:
            with open(filename, "rb") as file:
                self.image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                self.image = cv2.resize(self.image, (256, 256), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite("gris.tiff", self.image)
                self.mostrar_imagen()
    def abrir_marca(self):
        self.marca = None
        filename, _ = QFileDialog.getOpenFileName(None, 'Buscar Imagen', '.', 'Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)')
        if filename:
            with open(filename, "rb") as file:
                self.marca = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                self.marca = cv2.resize(self.marca, (32, 32), interpolation=cv2.INTER_CUBIC)
                ret, binary = cv2.threshold(self.marca, 127, 255, cv2.THRESH_BINARY)
                self.marca = binary
                cv2.imwrite("marcau.jpg",self.marca)
                self.mostrar_marca()

    def insertar_marca(self):
        self.logs.setPlainText("Insertando marca...")
        if self.image is not None:
            self.image = cv2.resize(self.image, (256,256), interpolation = cv2.INTER_CUBIC)
            self.image = self.image.astype(np.float64)
            '''Obtencion de los coeficientes de dc y los promedios
                para el entrenamiento de la red neuronal'''
            coef = []
            avg = []
            bloques = block.separa_bloque(self.image)
            i = 0
            while i < len(bloques):
                coef.append(block.dct_b(bloques[i]))
                i = i + 1
            # normalizar factores de dct
            max_dct = max(coef)
            i = 0
            while i < len(coef):
                coef[i] = coef[i] / max_dct
                i = i + 1
            i = 0
            while i < len(bloques):
                avg.append(block.average(bloques[i]))
                i = i + 1
            # normalizar datos del promedio
            self.max_avg = max(avg)
            i = 0
            while i < len(bloques):
                avg[i] = avg[i] / self.max_avg
                i = i + 1
            # entrenamiento de red neuronal
            self.logs.setPlainText(self.logs.toPlainText() + "\nEntrenando red neuronal")
            pat = [i for i in range(0, len(coef))]
            for i in range(0, len(coef)):
                pat[i] = [[coef[i]], [avg[i]]]

            salidas = []
            for i in range(0, len(pat)):
                s, network = block.return_output_nn(pat[i])
                salidas.append(s)
                self.net.append(network)
            # multiplicar promedio y salidas obtenidas por el valor maximo del promedio
            i = 0
            while i < len(bloques):
                avg[i] = avg[i] * self.max_avg
                i = i + 1
            i = 0
            while i < len(bloques):
                salidas[i] = salidas[i] * (self.max_avg)
                i = i + 1
            # incrustacion de la marca de agua
            suma_v = block.suma_bloque(bloques)

            img_marcada = np.zeros((self.image.shape), np.uint8)
            bloques_marcados = block.separa_bloque(img_marcada)
            lista_binaria = block.lista_marca(self.marca)
            lista_bloques_marcador = []
            q = 16.0
            for i in range(0, len(bloques_marcados)):
                bloque_m = bloques_marcados[i]
                bloque_o = bloques[i]
                if lista_binaria[i] == 255:
                    for x in range(bloque_m.shape[0]):
                        for y in range(bloque_m.shape[1]):
                            bloque_m[x, y] = bloque_o[x, y] + (
                                round(((0.25 * 8 * q + (64 * salidas[i]) - (suma_v[i]))) / suma_v[i]))
                else:
                    for x in range(bloque_m.shape[0]):
                        for y in range(bloque_m.shape[1]):
                            bloque_m[x, y] = bloque_o[x, y] - (
                                round(((0.25 * 8 * q + (64 * salidas[i]) - (suma_v[i]))) / suma_v[i]))
                lista_bloques_marcador.append(bloque_m)

            img_marcada = block.return_image(lista_bloques_marcador)
            self.logs.setPlainText(self.logs.toPlainText() + "\nPSNR: " + str(block.metricPSNR(img_marcada,self.image)))
            self.image = self.image.astype(np.uint8)
            self.image = img_marcada
            self.mostrar_imagen()
            self.save_image()

    def extraer_marca(self):
        self.abrir_imagen()
        bloques_img_marcada = block.separa_bloque(self.image)
        coef = []
        avg_n = []
        salidas_n = []
        i = 0
        while i < len(bloques_img_marcada):
            coef.append(block.dct_b(bloques_img_marcada[i]))
            i = i + 1
        max_dct = max(coef)
        i = 0
        while i < len(coef):
            coef[i] = coef[i] / max_dct
            i = i + 1
        for i in range(len(self.net)):
            c = []
            c.append(coef[i])
            salidas_n.append(self.net[i].runNN(c))
        sal = []
        i = 0
        while i < len(bloques_img_marcada):
            avg_n.append(block.average(bloques_img_marcada[i]))
            i = i + 1
        marca_extraida = []
        for i in salidas_n:
            sal.append(float(round(i[0] * (max(avg_n)))))
        for i in range(len(avg_n)):
            if avg_n[i] >= sal[i]:
                marca_extraida.append(255)
            else:
                marca_extraida.append(0)
        marca_extraida = block.return_marca(marca_extraida)
        cv2.imwrite("marca_extraida.png", marca_extraida)
        self.marca = marca_extraida
        self.logs.setPlainText("Se ha extraido la marca")
        self.mostrar_marca()

    def save_image(self):
        buttonReply = QMessageBox.question(self, 'Guardar imagen', "¿Deseas guardar la imagen marcada?",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            filename, _ = QFileDialog.getSaveFileName(None, 'Guardar Imagen')
            #cv2.imwrite(filename+".jpg",self.image,[cv2.IMWRITE_JPEG_QUALITY, 0])
            extension = filename.split(".")
            ex = str(extension[1])
            if ex == 'png':
                cv2.imwrite(filename, self. image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            elif ex == 'jpg':
                cv2.imwrite(filename, self.image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            else:
                cv2.imwrite(filename, self.image)

            self.logs.setPlainText(self.logs.toPlainText() + "\nImagen Guardada")
        else:
            self.logs.setPlainText(self.logs.toPlainText() + "\nNope :c")

    def mostrar_imagen(self):
        size = self.image.shape
        step = self.image.size / size[0]
        qformat = QImage.Format_Indexed8

        if len(size) == 3:
            if size[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.image, size[1], size[0], step, qformat)
        img = img.rgbSwapped()

        self.label.setPixmap(QPixmap.fromImage(img))
        self.resize(self.label.pixmap().size())
    def mostrar_marca(self):
        size = self.marca.shape
        step = self.marca.size / size[0]
        qformat = QImage.Format_Indexed8

        if len(size) == 3:
            if size[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.marca, size[1], size[0], step, qformat)
        img = img.rgbSwapped()

        self.label_marca.setPixmap(QPixmap.fromImage(img))
        self.resize(self.label_marca.pixmap().size())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = GuiWatermark()
    win.show()
    sys.exit(app.exec_())