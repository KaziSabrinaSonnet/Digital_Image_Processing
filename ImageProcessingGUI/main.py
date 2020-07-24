import sys
import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi



class ImageProcess(QDialog):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    def __init__(self):
        super(ImageProcess, self).__init__()
        loadUi('imageProcessing.ui', self)
        self.image = None
        self.referenceImage = None
        self.processedImage = None
        self.loadButton.clicked.connect(self.load)
        self.referenceButton.clicked.connect(self.load2)
        self.saveButton.clicked.connect(self.saveImage)
        self.cannyButton.clicked.connect(self.cannyClicked)
        self.hSlider.valueChanged.connect(self.cannyDisplay)
        self.detectionButton.clicked.connect(self.detectClicked)
        self.cartoonButton.clicked.connect(self.cartoonClicked)
        self.segmentButton.clicked.connect(self.segmentClicked)
        self.hismatchingButton.clicked.connect(self.histogram_match)
        self.hiseqButton.clicked.connect(self.histogram_equalization)
        self.bwButton.clicked.connect(self.blackAndwhite_Clicked)
        self.sketchButton.clicked.connect(self.sketchEffect)
        self.opButton.clicked.connect(self.oilPainting)

    @pyqtSlot()
    def cannyDisplay(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) >= 3 else self.image
        self.processedImage = cv2.Canny(gray_image, self.hSlider.value(), self.hSlider.value()*3)
        self.displayImage(3, 2)
    @pyqtSlot()
    def segmentClicked(self):
        image1 = self.image
        c = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        v = img.reshape((-1, 3))
        v = np.float32(v)
        it, label, center = cv2.kmeans(v, 4, None, c, 10, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        it = center[label.flatten()]
        self.processedImage = it.reshape((img.shape))
        self.displayImage(3, 2)
    @pyqtSlot()
    def histogram_match(self):
        from skimage.exposure import match_histograms
        self.processedImage = match_histograms(self.image, self.referenceImage, multichannel=True)
        self.displayImage(3, 2)
    @pyqtSlot()
    def oilPainting(self):
        img = self.image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        final_image = np.zeros(img.shape, np.uint8)
        for u in range(4, 420, 2):
            for v in range(4, 420, 2):
                box_element = [0, 0, 0]
                box_matrix = np.zeros(8, np.uint8)
                for w in range(-4, 4):
                    for x in range(-4, 4):
                        box_matrix[int(gray[u + w, v + x] / 32)] = box_matrix[int(gray[u + w, v + x] / 32)] + 1
                for w in range(-4, 4):
                    for x in range(-4, 4):
                        if int(gray[u + w, v + x] / 32) == np.argmax(box_matrix):
                            box_element = box_element + img[u + w, v + x]
                blue = int(box_element[0] / np.max(box_matrix))
                green = int(box_element[1] / np.max(box_matrix))
                red = int(box_element[2] / np.max(box_matrix))
                for w in range(2):
                    for x in range(2):
                        final_image[u + w, v + x] = (blue, green, red)
        self.processedImage = final_image
        self.displayImage(3, 2)
    @pyqtSlot()
    def blackAndwhite_Clicked(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) >= 3 else self.image
        (thresh, bwImage) = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        self.processedImage = bwImage
        self.displayImage(3, 2)
    @pyqtSlot()
    def histogram_equalization(self):
        img = self.image
        blue, green, red= cv2.split(img)
        hb, bb = np.histogram(blue.flatten(), 256, [0, 256])
        hg, bg = np.histogram(green.flatten(), 256, [0, 256])
        hr, br = np.histogram(red.flatten(), 256, [0, 256])
        cb = np.cumsum(hb)
        cg = np.cumsum(hg)
        cr = np.cumsum(hr)
        cb_mask = np.ma.masked_equal(cb, 0)
        cb_mask = (cb_mask - cb_mask.min()) * 255 / (cb_mask.max() - cb_mask.min())
        f1 = np.ma.filled(cb_mask, 0).astype('uint8')
        cg_mask = np.ma.masked_equal(cg, 0)
        cg_mask = (cg_mask - cg_mask.min()) * 255 / (cg_mask.max() - cg_mask.min())
        f2 = np.ma.filled(cg_mask, 0).astype('uint8')
        cr_mask = np.ma.masked_equal(cr, 0)
        cr_mask = (cr_mask - cr_mask.min()) * 255 / (cr_mask.max() - cr_mask.min())
        f3 = np.ma.filled(cr_mask, 0).astype('uint8')
        imb = f1[blue]
        img = f2[green]
        imr = f3[red]
        self.processedImage= cv2.merge((imb, img, imr))
        self.displayImage(3, 2)

    @pyqtSlot()
    def cartoonClicked(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) >= 3 else self.image
        gray_image = cv2.medianBlur(gray_image, 5)
        edge = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        image1 = cv2.bilateralFilter(self.image, 9, 300, 300)
        self.processedImage = cv2.bitwise_and(image1, image1, mask=edge)
        self.displayImage(3, 2)

    @pyqtSlot()
    def cannyClicked(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape)>=3 else self.image
        self.processedImage = cv2.Canny(gray_image, 100, 200)
        self.displayImage(3, 2)

    @pyqtSlot()
    def detectClicked(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) if len(self.image.shape) >= 3 else self.image
        faces = self.face_cascade.detectMultiScale(gray_image, 1.3, 5)
        for(point1, point2, width, height) in faces:
            if self.checkface.isChecked():
                cv2.rectangle(self.processedImage, (point1, point2), (point1+width, point2+height), (255, 0, 0), 2)
            else:
                self.processedImage = self.image.copy()
            face_gray = gray_image[point2:point2 + height, point1:point1 + width]
            face_color = self.processedImage[point2:point2 + height, point1:point1 + width]
            if self.checkeye.isChecked():
                eyes = self.eye_cascade.detectMultiScale(face_gray)
                for (point3, point4, width2, height2) in eyes:
                    cv2.rectangle(face_color, (point3, point4), (point3 + width2, point4 + height2), (0, 255, 0), 2)
            else:
                self.processedImage[point2:point2 + height, point1:point1 + height] = self.image[point2:point2 + height, point1:point1 + height].copy()

        self.displayImage(3, 2)
    @pyqtSlot()
    def sketchEffect(self):
        image= self.image
        gray_image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        negative_image = 255 - gray_image
        gaussian_blur = cv2.GaussianBlur(negative_image, ksize=(21, 21), sigmaX=0, sigmaY=0)
        self.processedImage= cv2.divide(gray_image, 255-gaussian_blur, scale=256)
        self.displayImage(3, 2)

    @pyqtSlot()
    def load(self):
        im, filter= QFileDialog.getOpenFileName(self, 'Open File', 'C:\\', "Image Files (*.jpg)")
        if im:
            self.loadImage(im)
        else:
            print("Invalid Image")

    @pyqtSlot()
    def load2(self):
        im1, filter1 = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\', "Image Files (*.jpg)")
        if im1:
            self.loadImage2(im1)
        else:
            print("Invalid Image")

    @pyqtSlot()
    def saveImage(self):
        im3, filter = QFileDialog.getSaveFileName(self, 'Save File', 'C:\\', "Image Files (*.jpg)")
        if im3:
            cv2.imwrite(im3, self.processedImage)
        else:
            print("Invalid Image")

    def loadImage(self,fname):
        self.image=cv2.imread(fname)
        self.processedImage = self.image.copy()
        self.displayImage(1, 1)

    def loadImage2(self, fname1):
        self.referenceImage = cv2.imread(fname1)
        self.displayImage(2,3)

    def displayImage(self, image_option=1, window= 1):
        qformat = QImage.Format_Indexed8
        imageToShow = None
        if image_option == 1:
            imageToShow = self.image
        elif image_option == 2:
            imageToShow = self.referenceImage
        elif image_option == 3:
            imageToShow = self.processedImage
        if len(imageToShow.shape) == 3:
            if(imageToShow.shape[2]) == 4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888
        img = QImage(imageToShow,imageToShow.shape[1], imageToShow.shape[0], imageToShow.strides[0],qformat)
        img = img.rgbSwapped()
        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        if window == 2:
            self.processedLabel.setPixmap(QPixmap.fromImage(img))
            self.processedLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        if window == 3:
            self.referencedLabel.setPixmap(QPixmap.fromImage(img))
            self.referencedLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


if __name__=='__main__':
    app=QApplication(sys.argv)
    window=ImageProcess()
    window.setWindowTitle('PyQt5 Image Processing GUI')
    window.show()
    sys.exit(app.exec_())