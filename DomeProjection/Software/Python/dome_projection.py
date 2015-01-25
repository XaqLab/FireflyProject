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
    mirror, and projector along with the geometry of the associated OpenGL
    virtual camera and screen.
    """
    def __init__(self,
                 screenWidth = 1,
                 screenHeight = 1,
                 camera2screenDist = 1,
                 imagePixelWidth = 512,
                 imagePixelHeight = 512,
                 projectorPixelWidth = 256,
                 projectorPixelHeight = 256):

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
        # Properties passed in as arguments
        ############################################################
        self._screenWidth = screenWidth
        self._screenHeight = screenHeight
        self._camera2screenDist = camera2screenDist
        self._imagePixelWidth = imagePixelWidth
        self._imagePixelHeight = imagePixelHeight
        self._projectorPixelWidth = projectorPixelWidth
        self._projectorPixelHeight = projectorPixelHeight

        ############################################################
        # Properties used to share results between method calls
        ############################################################
        self._projectorPixelRow = 0
        self._projectorPixelCol = 0

        ############################################################
        # Properties calculated from arguments
        ############################################################

        """
        Calculate the directions of all the OpenGL image pixels
        from the virtual camera's viewpoint.
        """

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

        # y is the distance from the camera to the screen
        y = self._camera2screenDist
        r = sqrt(x**2 + y**2 + z**2)
        self._cameraViewpoint = dstack((x/r, y/r, z/r))


        """
        Calculate the direction of all the projector pixel's projections
        from the mouse's viewpoint.
        """

        # Fake mapping for now, same as camera mapping
        rows = array([[float(i)]*self._projectorPixelWidth for i in 
                      range(self._projectorPixelHeight)])
        columns = array([[float(i)]*self._projectorPixelHeight for i in
                         range(self._projectorPixelWidth)]).T
        x = self._screenWidth*(columns/(self._projectorPixelWidth - 1) - 0.5)
        z = self._screenHeight*(rows/(self._projectorPixelHeight - 1) - 0.5)
        y = self._camera2screenDist
        r = sqrt(x**2 + y**2 + z**2)
        self._mouseViewpoint = dstack([x/r, y/r, z/r])

    
        """
        For each OpenGL image pixel use the directions calculated above
        to find the projector pixel with the closest direction.  Then add that
        OpenGL pixel to the projector pixel's list of contributing pixels.
        """

        # This 2D list of lists contains the list of OpenGL pixels
        # that contribute to each projector pixel.
        self._contributingPixels = \
            [[[] for i in range(self._projectorPixelHeight)]
             for j in range(self._projectorPixelWidth)]
        row = 0
        while row < self._imagePixelHeight:
            for col in range(self._imagePixelHeight):
                """
                For each OpenGL image pixel, determine which projector
                pixel has the closest direction.  
                """
                [r, c] = self._find_closest_projector_pixel(row, col) 
                self._contributingPixels[r][c].append([row, col])
            row = row + 1
            for col in range(self._imagePixelHeight - 1, -1, -1):
                """
                Go through the pixels in a serpentine pattern so that the
                current pixel is always close to the last pixel.  This way the
                search algorithm can use the last result as its starting point.
                """
                [r, c] = self._find_closest_projector_pixel(row, col) 
                self._contributingPixels[r][c].append([row, col])
            row = row + 1


    def _find_closest_projector_pixel(self, row, col):
        """
        For the OpenGL image pixel specified by row and col use the directions
        in self._comeraViewpoint and self._mouseViewpoint to find the projector
        pixel which has the closest direction and return its row and column in
        a list.  This is done using a search method rather than calculating the
        dot product for every projector pixel.
        """
        cameraDirection = self._cameraViewpoint[row, col]

        # Start with the last projector pixel
        r = self._projectorPixelRow
        c = self._projectorPixelCol
        mouseDirection = self._mouseViewpoint[r, c]

        # Calculate dot product of this OpenGL pixel
        # with the last projector pixel.
        dp = dot(cameraDirection, mouseDirection)

        # Calculate dot product of this OpenGL pixel with the
        # neighboring projector pixels.
        [neighbor_dps, neighbors] = \
            self._calc_neighbor_dot_products(r, c, cameraDirection)

        while max(neighbor_dps) > dp:
            """
            If the dot product with one of the neighboring projector pixels is 
            larger then update r and c to that pixel and check its neighbors.
            Repeat until all neighbors have smaller (or equal) dot products.
            """
            dp = max(neighbor_dps)
            [r, c] = neighbors[neighbor_dps.index(dp)]
            [neighbor_dps, neighbors] = \
                self._calc_neighbor_dot_products(r, c, cameraDirection)

        # Save projector pixel for next call
        self._projectorPixelRow = r
        self._projectorPixelCol = c
        return [r, c]


    def _calc_neighbor_dot_products(self, row, col, cameraDirection):
        """
        For the projector pixel specified by row and col, calculate the dot
        product of all its neighbors with the given camera direction.  Return
        a list containing a list of the dot products and a list of the row and
        column for each corresponding pixel.
        """
        neighbors = []
        neighbor_dps = []

        # find neighbors
        row_above = [[-1, -1], [-1, 0], [-1, 1]]
        beside = [[0, -1], [0, 1]]
        row_below = [[1, -1], [1, 0], [1, 1]]
        offsets = row_above + beside + row_below
        for [dr, dc] in offsets:
            if row + dr >= 0 and row + dr < self._projectorPixelHeight:
                if col + dc >= 0 and col + dc < self._projectorPixelWidth:
                    neighbors.append([row + dr, col + dc])

        # calculate neighbor dot products
        for neighbor in neighbors:
            neighborDirection = self._mouseViewpoint[neighbor[0], neighbor[1]]
            neighbor_dps.append(dot(cameraDirection, neighborDirection))

        return [neighbor_dps, neighbors]


    def warpImageForDome(self, image):
        """
        Take an RGB input image intended for display on a flat screen and
        produce an image for the projector that removes the distortions caused
        by projecting the image onto the dome using a spherical mirror.
        """
        assert image.size == (self._imagePixelWidth, self._imagePixelHeight)

        pixels = array(image)
        warpedPixels = zeros([image.size[0], image.size[0], 3], dtype=uint8)
        for row in range(self._projectorPixelHeight):
            for col in range(self._projectorPixelWidth):
                pixelValue = zeros(3)
                for n, pixel in enumerate(self._contributingPixels[row][col]):
                    pixelValue += pixels[pixel[0], pixel[1]]
                pixelValue = pixelValue/(n + 1)
                warpedPixels[row][col] = array(pixelValue, dtype=uint8)

        return Image.fromarray(warpedPixels, mode='RGB')



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
    #[rows, columns] = input_image.size
    #pixels = array(input_image)
    print input_image.size
else:
    print "Unsupported image mode:", input_image.mode
    exit()

# show original image
input_image.show()

print "Creating instance of domeDisplay Class"
dome = domeDisplay()
print "Done initializing dome"

#output_image = Image.fromarray(pixels, mode='RGB')
output_image = dome.warpImageForDome(input_image)

output_image.show()
#output_image.save("output_file", "PNG")


