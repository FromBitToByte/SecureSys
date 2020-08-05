import  numpy as np
import cv2
import imageio
import random
import decimal
import math
import pickle
import  dill
#------------------------------------------------------------------------------------------------------------------------------------------------------
getIfromRGB = lambda val : np.int32( (val[0]<<16) + (val[1]<<8) + val[2]) #this lamda expression converts pixel value(rgb) to int
getRGBfromI = lambda val : [ (val>>16)&255, (val>>8)&255, val&255 ] #this function gives rgb value from int
#------------------------------------------------------------------------------------------------------------------------------------------------------
class Key:
    def __init__(self):#constructor
        self.count = 0
        self.coordinates = np.zeros( ( 0, 4 ), dtype='i' ) #coordinates of faces (x, y, w, h)
        self.n_seg = np.zeros( shape=(0), dtype='i') # no of segments in which each pix array will be divided
        self.lm = np.zeros( shape=(0, 2), dtype=decimal.Decimal ) # initial values of logistic map for each face (lamda, sl0)
        self.sm = np.zeros( shape=(0, 2), dtype=decimal.Decimal ) # initial values of sine map for each face (sigma, xs0 )

#------------------------------------------------------------------------------------------------------------------------------------------------------
class Decrypt: #this class is used to Decrypt the image

    def __init__( self , fn ): #constructor
        self.fileName = fn;
        # self.im = imageio.imread( "images_generated" + '\\' + fn )
        self.im = cv2.imread("embeeded_images" + '\\' + fn)
        self.key = None #key generated after encryption

    # --------------------------------------------------------------------------------------------------------------------
    def extract(self, x, y, w, h ):
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
    def getBackOriginal( self, index,  pix ):
        ret = np.copy(pix)
        l = self.key.lm[index][0]
        s0 = self.key.lm[index][1]
        i = 0
        scurr = s0
        for val in ret:
            log = l * scurr * (1 - scurr)
            scurr = log
            ret[i] = val ^ int(round(log * 16777215))
            i += 1
        return ret
    # --------------------------------------------------------------------------------------------------------------------
    def assemble(self, index, pix):  # step 2 of encryption (scrambling of pixels is done here)
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
        return  ret
    # --------------------------------------------------------------------------------------------------------------------
    def readKey(self):
        fn = self.fileName.split('.', 1)[0]
        rk = open("extracted_keys\\" + fn + "_key.txt", 'rb')
        self.key = pickle.load( rk )
        # self.key = dill.load( rk )
        rk.close()


    # --------------------------------------------------------------------------------------------------------------------
    def reassamble(self, pix, x, y, w, h):
        indx = 0
        j = 0
        while j < h:
            i = 0
            while i < w:
                self.im[y + i][x + j] = getRGBfromI(pix[indx])
                indx += 1
                i += 1
            j += 1
    # --------------------------------------------------------------------------------------------------------------------
    def decrypt( self ):
        self.readKey()
        i = self.key.count-1;
        while i >= 0:
            pix = self.extract( self.key.coordinates[i][0], self.key.coordinates[i][1], self.key.coordinates[i][2], self.key.coordinates[i][3] )
            pix = self.assemble(  i, pix )
            pix = self.getBackOriginal( i, pix)
            self.reassamble( pix, self.key.coordinates[i][0], self.key.coordinates[i][1], self.key.coordinates[i][2],self.key.coordinates[i][3])
            i -= 1
        cv2.imwrite("decrypted_images\\" + self.fileName,self.im)  # encrypted image is written in images_generated file
    # --------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------------------------------
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

def compareImages( fileName ):
    original_img = cv2.imread("embeeded_images" + '\\' + fileName, 1)
    encrypted_img = cv2.imread("decrypted_images"+ '\\' + fileName, 1)
    original_img = ResizeWithAspectRatio( original_img, width=500)
    encrypted_img = ResizeWithAspectRatio(encrypted_img, width=500)

    h_stack = np.hstack( (original_img,encrypted_img) )

    cv2.imshow( "Encrypted and Decrypted Images : ", h_stack )

    cv2.waitKey()


#------------------------------------------------------------------------------------------------------------------------------------------------------
def main():
    # fileName = "image3.png"
    fileName = "image1.png"

    #create object
    decryptor = Decrypt( fileName )
    decryptor.decrypt()

    compareImages(fileName)

    print( "Decryption Done :)" )
#------------------------------------------------------------------------------------------------------------------------------------------------------
main()
#------------------------------------------------------------------------------------------------------------------------------------------------------
