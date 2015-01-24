#!/usr/local/bin/python
# written for python 2.7.8

"""
This is the mapping to project images from OpenGL onto the hemispherical dome
display for mouse virtual reality experiments.  The objective of this mapping
is to ensure that the OpenGL image and the image on the dome display look the
same from the viewer's point of view.  For the dome display the viewer is the
mouse and for the OpenGL image the viewer is a virtual camera.  Each projector
pixel projects to a small region seen by the mouse.  The RGB values for each
projector pixel are calculated from pixels in the OpenGL image such that the
pixels in the OpenGL image and the corresponding region seen by the mouse are
in the same direction.  A mapping from projector pixels to the mouse's
viewpoint is used to determine which pixels in the OpenGL image contribute to
each projector pixel.  The RGB values for each projector pixel are the 
average of the RGB values from the OpenGL image contributing pixels.
"""

class domeDisplay:
    """
    The dome display class describes the geometry of our dome, spherical
    mirror, and projector as well as the geometry of OpenGL's virtual camera
    and screen.
    """
    def __init__(self,
                 screenWidth = 1,
                 screenHeight = 1,
                 camera2screenDist = 1,
                 imagePixelWidth = 512,
                 imagePixelHeight = 512,
                 projectorPixelWidth = 512,
                 projectorPixelHeight = 512):

        """
        Parameters:
        ----------------------------
        @param screenWidth: 
            The width of the OpenGL screen in arbitrary units.
        @param screenHeight: 
            The height of the OpenGL screen in arbitrary units.
        @param camera2screenDist: 
            The distance from OpenGL's virtual camera to the OpenGL screen in
            arbitrary units.
        @param imagePixelWidth:
            The number of horizontal pixels in the OpenGL image.
        @param imagePixelHeight:
            The number of vertical pixels in the OpenGL image.
        @param projectorPixelWidth:
            The number of horizontal pixels in the projector image.
        @param projectorPixelHeight:
            The number of vertical pixels in the projector image.
        """

        ############################################################
        # properties passed in as arguments
        ############################################################
        self._screenWidth = screenWidth
        self._screenHeight = screenHeight
        self._camera2screenDist = camera2screenDist
        self._imagePixelWidth = imagePixelWidth
        self._imagePixelHeight = imagePixelHeight
        self._projectorPixelWidth = projectorPixelWidth
        self._projectorPixelHeight = projectorPixelHeight

        ############################################################
        # Properties calculated from arguments
        ############################################################

        ############################################################
        # Calculate the directions of all the OpenGL image pixels
        # from the virtual camera's viewpoint.
        ############################################################

        # Make matricies of row and column values
        rows = array([[float(i)]*self._imagePixelWidth for i in 
                      range(self._imagePixelHeight)])
        columns = array([[float(i)]*self._imagePixelHeight for i in
                         range(self._imagePixelWidth)]).T

        # Calculate x and z values from column and row values so they
        # are symmetric about the center of the image and scaled to the
        # screen size.
        x = self._screenWidth*(columns/(self._imagePixelWidth - 1) - 0.5)
        z = self._screenHeight*(rows/(self._imagePixelHeight - 1) - 0.5)

        debug_image = Image.fromarray(255*(x + 0.5), mode='L')
        debug_image.show()

        # y is the distance from the camera to the screen
        y = self._camera2screenDist
        r = sqrt(x**2 + y**2 + z**2)
        self._cameraViewpoint = dstack((x/r, y/r, z/r))


        ############################################################
        # Calculate the direction of all the projector pixel's
        # projections from the mouse's viewpoint.
        ############################################################

        # Fake mapping for now, same as camera mapping
        rows = array([[float(i)]*self._imagePixelWidth for i in 
                      range(self._imagePixelHeight)])
        columns = array([[float(i)]*self._imagePixelHeight for i in
                         range(self._imagePixelWidth)]).T
        x = self._screenWidth*(columns/(self._imagePixelWidth - 1) - 0.5)
        z = self._screenHeight*(rows/(self._imagePixelHeight - 1) - 0.5)
        y = self._camera2screenDist
        r = sqrt(x**2 + y**2 + z**2)
        self._mouseViewpoint = dstack([x/r, y/r, z/r])

    
        ############################################################
        # Calculate the weights used to compute the projector pixel
        # values from the OpenGL image's pixel values.
        ############################################################

        """
        Calculate the binary weight matrix used to calculate projector pixel
        RGB values from OpenGL image pixel RGB values.  For each OpenGL image
        pixel determine its direction as measured from the virtual camera.
        Each OpenGL image pixel will contribute to the the projector pixel to
        which it is closest in direction.
        """
        # make a vector out of the virtual camera viewpoint
        numImagePixels = self._imagePixelHeight * self._imagePixelWidth
        imageVector = reshape(self._cameraViewpoint, [numImagePixels, 3])

        # make a vector out of the mouse's viewpoint
        numProjPixels = self._projectorPixelHeight * self._projectorPixelWidth
        projectorVector = reshape(self._mouseViewpoint, [numProjPixels, 3])

        # Calculate weights
        # Need to calculate outer products of inner products
        self._weights[1] = outer(imageVector, projectorVector)



############################################################
# Main Program Starts Here
############################################################
from numpy import *
import matplotlib
import matplotlib.pyplot as plot
from PIL import Image

input_image = Image.open("Lenna.png")

# read pixel data from image into a list
if input_image.mode == 'RGB':
    print "RGB image"
    [rows, columns] = input_image.size
    pixels = array(input_image)
    print shape(pixels)
else:
    print "Unsupported image mode:", input_image.mode
    exit()

# show original image
input_image.show()

dome = domeDisplay()

#for col in range(1280):
    #for row in range(720):
        #""" for each projector pixel (1280x720) """

        #""" compute it's RGB (or YUV) values as a weighted sum of the values for pixels
        #in the OpenGL image """
        #projector_pixel[x][y] = weights * open_gl_image

for row in range(rows):
    #for col in range(columns):
    for col in range(10):
        pixels[row][col] = (0, 0, 0)


output_image = Image.fromarray(pixels, mode='RGB')

output_image.show()
#output_image.save("output_file", "PNG")


