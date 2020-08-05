import numpy as np
import cv2
import imageio
import random
import decimal
import math

getIfromRGB = lambda val: int((val[0] << 16) + (val[1] << 8) + val[2])
getRGBfromI = lambda val: [(val >> 16) & 255, (val >> 8) & 255, val & 255]


# ------------------------------------------------------------------------------------------------------------------------
class faceDetector:  # this class will be used to get the cooridinates of all the faces in Image
    def __init__(self, imageName):
        self.fileName = imageName

    def getFaceCoordinates(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # open the image for detection
        img = cv2.imread("images_original" + '\\' + self.fileName, 1)

        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)

        return faces


# ------------------------------------------------------------------------------------------------------------------------
class Key:
    def __init__(self):
        self.count = 0
        self.coordinates = np.zeros((0, 4), dtype='i')  # coordinates of faces (x, y, w, h)
        self.n_seg = np.zeros(shape=(0), dtype='i')  # no of segments in which each pix array will be divided
        self.lm = np.zeros(shape=(0, 2),
                           dtype=decimal.Decimal)  # initial values of logistic map for each face (lamda, sl0)
        self.sm = np.zeros(shape=(0, 2),
                           dtype=decimal.Decimal)  # initial values of sine map for each face (sigma, xs0 )


class Encrypt:  # this class is used to encrypt the image

    def __init__(self, fc, fn):
        self.faceCoordinate = fc
        self.fileName = fn;
        self.im = imageio.imread("images_original" + '\\' + fn)
        self.key = Key()  # this is the key that will be generated after encrupting process and will be written in a file

    def writeKey(self):
        print("Generated Key is written in Key Folder")

    def extract(self, x, y, w, h):
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

    def confuse(self, pix):  # xor is performed here
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

    def diffuse(self, pix):  # scrambling of pixels is done here
        # print( pix )
        ret = np.copy(pix)
        n = int(ret.shape[0])
        n_seg = random.randrange(10, int(n / 100))
        sig = decimal.Decimal(random.randrange(8700000, 10000000)) / 10000000
        x0 = decimal.Decimal(random.randrange(0, 10000000)) / 10000000
        spix = int(math.ceil(n / n_seg))
        self.key.n_seg = np.append(self.key.n_seg, n_seg)  # adding n_seg to key
        self.key.sm = np.append(self.key.sm, [[sig, x0]], axis=0)
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
                xcurr = val_s
                position = round(val_s * decimal.Decimal(len(pos) - 1))
                list1.append(pos[position])
                pos.remove(pos[position])
                i += 1
                indx += 1
            i = 0
            for ok in list1:
                ret[start + i] = pix[start + ok]
                i += 1
            num_seg += 1

        # print(ret)
        # print(pix)
        # print('\n')
        return ret

    def reassamble(self, pix, x, y, w, h):  # the encrypted values are imbeeded in self.im
        indx = 0
        j = 0
        while j < h:
            i = 0
            while i < w:
                print( self.im[y+i][x+j], end = ' ' )
                self.im[y + i][x + j] = getRGBfromI(pix[indx])
                print(self.im[y + i][x + j])
                indx += 1
                i += 1
            j += 1

    def decrypt(self, index, pix):
        ret = np.copy(pix)
        n = int(ret.shape[0])
        n_seg = (self.key.n_seg[index])
        sig = self.key.sm[index][0]
        x0 = self.key.sm[index][1]
        spix = int(math.ceil(n / n_seg))

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
                xcurr = val_s
                position = round(val_s * decimal.Decimal(len(pos) - 1))
                list1.append(pos[position])
                pos.remove(pos[position])
                i += 1
                indx += 1

            i = 0
            for ok in list1:
                ret[start + ok] = pix[start + i]
                i += 1
            num_seg += 1

        l = self.key.lm[index][0]
        s0 = self.key.lm[index][1]
        i = 0
        scurr = s0
        for val in ret:
            log = l * scurr * (1 - scurr)
            scurr = log
            ret[i] = val ^ int(round(log * 16777215))
            i += 1
        # print( ret )
        return ret

    def encrpyt(self):
        for (x, y, w, h) in self.faceCoordinate:
            self.key.count += 1
            pix = self.extract(x, y, w, h)
            pix = self.confuse(pix)
            pix = self.diffuse(pix)
            self.reassamble(pix, x, y, w, h)
        imageio.imwrite("images_generated\\" + self.fileName,
                        self.im)  # encrypted image is written in images_generated file
        # self.im = imageio.imread("images_generated\\" + self.fileName)
        i = 0
        while i < self.key.count:
            pix1 = self.extract(self.key.coordinates[i][0], self.key.coordinates[i][1], self.key.coordinates[i][2],
                                self.key.coordinates[i][3])
            pix1 = self.decrypt(i, pix1)
            self.reassamble(pix1, self.key.coordinates[i][0], self.key.coordinates[i][1], self.key.coordinates[i][2],
                            self.key.coordinates[i][3])
            i += 1
        imageio.imwrite("decrypted_images\\" + self.fileName,
                        self.im)  # encrypted image is written in images_generated file


# ------------------------------------------------------------------------------------------------------------------------

def main():
    fileName = "image1.jpg"
    # step = 1
    # Detect Faces in Image
    obj = faceDetector(fileName)
    faceCooridingates = obj.getFaceCoordinates()  # a numpy 3d array

    # step 2
    # Encrypt Faces
    encryptor = Encrypt(faceCooridingates, fileName)
    encryptor.encrpyt()
    print("Done")


main()
