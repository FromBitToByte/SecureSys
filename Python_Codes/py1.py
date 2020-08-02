import  numpy as np
import cv2
import imageio
#------------------------------------------------------------------------------------------------------------------------
class faceDetector: #this class will be used to get the cooridinates of all the faces in Image
    def __init__(self, imageName):
        self.fileName = imageName

    def getFaceCoordinates( self ):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # open the image for detection
        img = cv2.imread("images_original" + '\\' + self.fileName, 1)

        # Convert into grayscale
        gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale( gray, 1.3, 4)

        return faces
#------------------------------------------------------------------------------------------------------------------------

class Encrypt : #this class is used to encrypt the image

    def __init__( self, fc, fn ):
        self.faceCoordinate = fc
        self.fileName = fn;
        self.im = imageio.imread( "images_original" + '\\' + fn )
        self.Key = None #this is the key that will be generated after encrupting process and will be written in a file

    def confuse(self): # xor is performed here
        print( "Insert code to confuse here " )

    def diffuse(self): #scrambling of pixels is done here
        print("Insert code to diffuse here ")

    def reassamble(self): #the encrypted values are imbeeded in self.im
        print("Insert code to reassamble here ")
    def writeKey(self):
        print("Generated Key is written in Key Folder" )

    def encrpyt(self):
        self.confuse()
        self.diffuse()
        self.reassamble()
        self.writeKey()
        imageio.imwrite("images_generated\\" + self.fileName, self.im) #encrypted image is written in images_generated file

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

#------------------------------------------------------------------------------------------------------------------------

def main():
    fileName = "image1.jpg"
    #step = 1
    #Detect Faces in Image
    obj = faceDetector( fileName )
    faceCooridingates = obj.getFaceCoordinates() # a numpy 3d array

    #step 2
    #Encrypt Faces
    encryptor = Encrypt( faceCooridingates,  fileName )
    encryptor.encrpyt()

    print( "Done" )

main()
