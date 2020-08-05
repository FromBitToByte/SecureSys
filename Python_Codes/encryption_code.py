import numpy as np
import cv2
import imageio
import random
import decimal
import math
import pickle
import dill

# ------------------------------------------------------------------------------------------------------------------------------------------------------
getIfromRGB = lambda val: int(
    (val[0] << 16) + (val[1] << 8) + val[2])  # this lamda expression converts pixel value(rgb) to int
getRGBfromI = lambda val: [(val >> 16) & 255, (val >> 8) & 255, val & 255]  # this function gives rgb value from int


# ------------------------------------------------------------------------------------------------------------------------------------------------------
class faceDetector:  # this class will be used to get the cooridinates of all the faces in Image
    def __init__(self, imageName):
        self.fileName = imageName

    def getFaceCoordinates(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # open the image for detection
        img = cv2.imread("original_images" + '\\' + self.fileName, 1)

        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)

        return faces


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# during encryption key will be generated, generated key will be an object of Key class
class Key:
    def __init__(self):  # constructor
        self.count = 0
        self.coordinates = np.zeros((0, 4), dtype='i')  # coordinates of faces (x, y, w, h)
        self.n_seg = np.zeros(shape=(0), dtype='i')  # no of segments in which each pix array will be divided
        self.lm = np.zeros(shape=(0, 2),
                           dtype=decimal.Decimal)  # initial values of logistic map for each face (lamda, sl0)
        self.sm = np.zeros(shape=(0, 2),
                           dtype=decimal.Decimal)  # initial values of sine map for each face (sigma, xs0 )


# ------------------------------------------------------------------------------------------------------------------------------------------------------
class Encrypt:  # this class is used to encrypt the image

    def __init__(self, fc, fn):  # constructor
        self.faceCoordinate = fc
        self.fileName = fn;
        # self.im = imageio.imread( "images_original" + '\\' + fn )
        self.im = cv2.imread("original_images" + '\\' + fn)
        self.key = Key()  # this is the key that will be generated after encrupting process and will be written in a file

    # --------------------------------------------------------------------------------------------------------------------
    def extract(self, x, y, w,
                h):  # this function is used to extract pixels of face from self.im variable and returns an numpy array of int
        self.key.coordinates = np.append(self.key.coordinates, [[x, y, w, h]],
                                         axis=0)  # adding face coordinates into key
        pix = np.zeros(shape=(0, 1), dtype='i')
        j = 0
        while j < h:
            i = 0
            while i < w:
                pix = np.append(pix, getIfromRGB(self.im[y + i][x + j]))
                i += 1
            j += 1
        return pix

    # --------------------------------------------------------------------------------------------------------------------
    def confuse(self, pix):  # step1 of encryption ( xor operation is performed here )
        l = decimal.Decimal(random.randrange(3560000, 4000000)) / 1000000
        s0 = decimal.Decimal(random.randrange(0, 1000000)) / 1000000
        self.key.lm = np.append(self.key.lm, [[l, s0]], axis=0)
        ret = pix
        i = 0
        scurr = s0
        for val in pix:
            log = l * scurr * (1 - scurr)
            scurr = log
            ret[i] = val ^ int(round(log * 16777215))
            i += 1
        return ret

    # --------------------------------------------------------------------------------------------------------------------
    def diffuse(self, pix):  # step 2 of encryption (scrambling of pixels is done here)
        ret = np.copy(pix)
        n = int(ret.shape[0])

        # generating initial values for encryption
        n_seg = random.randrange(10, int(n / 100))
        sig = decimal.Decimal(random.randrange(8700000, 10000000)) / 10000000
        x0 = decimal.Decimal(random.randrange(0, 10000000)) / 10000000
        spix = int(math.ceil(n / n_seg))

        self.key.n_seg = np.append(self.key.n_seg, n_seg)  # adding n_seg to key
        self.key.sm = np.append(self.key.sm, [[sig, x0]], axis=0)  # adding values to key

        xcurr = x0
        indx = 0
        num_seg = 0

        while num_seg < n_seg:
            start = indx

            ma = min(n, indx + spix) - indx  # size of pix array
            pos = [ok for ok in range(ma)]

            # scrambling is done here
            i = 0
            list1 = []
            while i < ma:
                val_s = sig * decimal.Decimal(math.sin(decimal.Decimal(math.pi) * xcurr))
                position = round(val_s * decimal.Decimal(len(pos) - 1))

                list1.append(pos[position])
                pos.remove(pos[position])

                xcurr = val_s
                i += 1
                indx += 1

            i = 0
            for ok in list1:
                ret[start + i] = pix[start + ok]
                i += 1
            num_seg += 1

        return ret

    # --------------------------------------------------------------------------------------------------------------------
    def writeKey(self):
        fn = self.fileName.split('.', 1)[0]
        wk = open("generated_keys\\" + fn + "_key.txt", 'wb')
        pickle.dump(self.key, wk)
        # dill.dump(self.key, wk )
        print(self.key.count)
        print( self.key.n_seg )
        print(self.key.lm)
        print(self.key.sm)

        wk.close()

    # --------------------------------------------------------------------------------------------------------------------
    def reassamble(self, pix, x, y, w, h):  # the encrypted values are imbeeded in self.im
        indx = 0
        j = 0
        while j < h:
            i = 0
            while i < w:
                self.im[y + i][x + j] = getRGBfromI(pix[indx])
                indx += 1
                i += 1
            # break;
            j += 1

    # --------------------------------------------------------------------------------------------------------------------
    def encrpyt(self):
        for (x, y, w, h) in self.faceCoordinate:
            self.key.count += 1
            pix = self.extract(x, y, w, h)
            pix = self.confuse(pix)
            pix = self.diffuse(pix)
            self.reassamble(pix, x, y, w, h)

        self.writeKey()  # Generated key is written
        # imageio.imwrite("generated_images\\" + self.fileName, self.im) #encrypted image is written in images_generated folder
        cv2.imwrite("generated_images\\" + self.fileName,
                    self.im)  # encrypted image is written in images_generated folder

    # --------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------------------------------------------
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def compareImages(fileName):
    original_img = cv2.imread("original_images" + '\\' + fileName, 1)
    encrypted_img = cv2.imread("generated_images" + '\\' + fileName, 1)
    original_img = ResizeWithAspectRatio(original_img, width=500)
    encrypted_img = ResizeWithAspectRatio(encrypted_img, width=500)

    h_stack = np.hstack((original_img, encrypted_img))

    cv2.imshow("Original and Encrypted Images : ", h_stack)

    cv2.waitKey()


# ------------------------------------------------------------------------------------------------------------------------------------------------------
def main():
    # fileName = "image1.png"
    fileName = "image1.png"

    # step = 1
    # Detect Faces in Image
    obj = faceDetector(fileName)
    faceCooridingates = obj.getFaceCoordinates()  # a numpy 3d array

    # step 2
    # Encrypt Faces
    encryptor = Encrypt(faceCooridingates, fileName)
    encryptor.encrpyt()

    # step3
    # render original image and encrypted image
    compareImages(fileName)

    print("Face Encryption Done Successfully :) ")


# ------------------------------------------------------------------------------------------------------------------------------------------------------
main()
# ------------------------------------------------------------------------------------------------------------------------------------------------------
