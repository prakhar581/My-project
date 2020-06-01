# PossibleChar.py
# AIM: to find width, height, centre_x ,centre_y, Diagonal and aspect ration and ue it latter
import cv2
import numpy as np
import math

###################################################################################################
class PossibleChar:

    # constructor #################################################################################
    # It will be called as soon as class object is created
    def __init__(self, contour):
        self.contour = contour #contour is a method

        #self.boundingRect = cv2.boundingRect(self.contour) # getting bounding rect and saving to boundingRect 

        #[intX, intY, intWidth, intHeight] = self.boundingRect 

       # self.intBoundingRectX = intX
       # self.intBoundingRectY = intY
       # self.intBoundingRectWidth = intWidth
        #self.intBoundingRectHeight = intHeight
        self.intBoundingRectX,self.intBoundingRectY,self.intBoundingRectWidth,self.intBoundingRectHeight=cv2.boundingRect(self.contour)
        self.intBoundingRectArea = self.intBoundingRectWidth * self.intBoundingRectHeight

        self.intCenterX = (self.intBoundingRectX + self.intBoundingRectX + self.intBoundingRectWidth) / 2
        self.intCenterY = (self.intBoundingRectY + self.intBoundingRectY + self.intBoundingRectHeight) / 2
        
        self.fltDiagonalSize = math.sqrt((self.intBoundingRectWidth ** 2) + (self.intBoundingRectHeight ** 2))#sqrt(x2+y2)

        self.fltAspectRatio = float(self.intBoundingRectWidth) / float(self.intBoundingRectHeight)
        # aspect ration is width/height
    # end constructor

# end class








