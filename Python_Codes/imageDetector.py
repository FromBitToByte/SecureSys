import numpy as np
import cv2
import imageio


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

        # print( type(faces) )

        # Draw rectangle around the faces
        # for (x, y, w, h) in faces :
        #     cv2.rectangle( img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Display the output
        # cv2.imshow('image', img)

        # cv2.imwrite("images_generated\\" + self.fileName, img)
        # cv2.waitKey()

        return faces


# ------------------------------------------------------------------------------------------------------------------------

class Encrypt:

    def __init__(self, fc, fn):
        self.faceCoordinate = fc
        self.fileName = fn;
        self.im = imageio.imread("images_original" + '\\' + fn)
        selfKey = None

    def confuse(self):
        print("Insert code to confuse here ")

    def diffuse(self):
        print("Insert code to diffuse here ")

    def reassamble(self):
        print("Insert code to reassamble here ")

    def encrpyt(self):
        self.confuse()
        self.diffuse()
        self.reassamble()
        imageio.imwrite("images_generated\\" + self.fileName, self.im)
        # for ( x, y, w, h ) in self.faceCoordinate:
        #     i = 0
        #     while i < h :
        #         j = 0
        #         while j < w:
        #             self.im[y+j][x+i] = [0, 0, 0 ]
        #             j += 1
        #         i += 1
        #
        #
        # imageio.imwrite("images_generated\\" + self.fileName, self.im)


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
