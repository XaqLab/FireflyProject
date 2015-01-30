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

DEBUG = True

class domeDisplay:
    """
    The dome display class describes the geometry of our dome, spherical
    mirror, and projector along with the geometry of the associated OpenGL
    virtual camera and screen.
    """
    def __init__(self,
                 screenWidth = 1,
                 screenHeight = 1,
                 camera2screenDist = 0.5,
                 imagePixelWidth = 512,
                 imagePixelHeight = 512,
                 projectorPixelWidth = 512,
                 projectorPixelHeight = 512,
                 firstProjectorImage = [[-0.09, 0.43, 0.18], [0.09, 0.43, 0.18], 
                                        [0.09, 0.43, 0.04], [-0.09, 0.43, 0.04]],
                 secondProjectorImage = [[-0.11, 0.36, 0.21], [0.11, 0.36, 0.21],
                                         [0.11, 0.36, 0.04], [-0.11, 0.36, 0.04]],
                 mirrorRadius = 0.2286,
                 domeCenter = [0, 0.14, 0.42],
                 domeRadius = 0.64,
                 mousePosition = [0, 0, 0.5]
                 ):

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
        @param firstProjectorImage:
            A list of four (x,y1,z) points, starting top left and proceeding
            clockwise, that specifies the corners of the projector's image
            at a distance y1 from the center of the mirror.
        @param secondProjectorImage:
            A list of four (x,y2,z) points, starting top left and proceeding
            clockwise, that specifies the corners of the projector's image
            at a distance y2 from the center of the mirror.
        @param mirrorRadius:
            The radius of the mirror in arbitrary units.
        @param domeCenter:
            An (x,y,z) vector from the center of the mirror to the center of
            the dome.
        @param domeRadius:
            The radius of the dome in arbitrary units.
        @param mousePosition:
            An (x,y,z) vector from the center of the mirror to the position
            of the mouse's eyes.
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
        self._firstProjectorImage = firstProjectorImage
        self._secondProjectorImage = secondProjectorImage

        ############################################################
        # Properties used to share results between method calls
        ############################################################

        # Start search low in the middle of the projector image 
        self._projectorPixelRow = 3*projectorPixelHeight/4
        self._projectorPixelCol = projectorPixelWidth/2

        ############################################################
        # Properties calculated from arguments
        ############################################################

        ############################################################
        # Calculate the directions of all the OpenGL image pixels
        # from the virtual camera's viewpoint.
        ############################################################

        """
        All positions are relative to OpenGL's virtual camera which
        is looking down the positive y-axis.
        """
        self._cameraViewDirections = \
            flatDisplayDirections(screenHeight, screenWidth,
                                  imagePixelHeight, imagePixelWidth,
                                  camera2screenDist)


        ##################################################################
        # Calculate the direction of all the projector pixel's projections
        # from the mouse's viewpoint.
        ##################################################################

        """
        All positions are relative to the center of the hemispherical mirror.
        The projector is on the positive y-axis (but projecting in the
        -y direction) and its projected image is assumed to be horizontally
        centered on the mirror.
        """

        # Find the position of the projector's focal point.
        self._projectorPosition = self._find_projector_focal_point()


        """
        Calculate the directions from projector's focal point for each
        projector pixel.  This calculation assumes secondProjectorImage has
        the same width at the top and bottom.
        """
        imageWidth = secondProjectorImage[1][0] - secondProjectorImage[0][0]
        imageHeight = secondProjectorImage[0][2] - secondProjectorImage[2][2]
        distanceToImage = (secondProjectorImage[0][1]
                           - self._projectorPosition[1])
        verticalOffset = secondProjectorImage[2][2] + imageHeight/2
        self._projectorPixelDirections = \
            flatDisplayDirections(imageHeight, imageWidth,
                                  projectorPixelHeight, projectorPixelWidth,
                                  distanceToImage, verticalOffset)

        # Flip the sign of the x-values because projection is in -y direction
        self._projectorPixelDirections *= array([-1, 1, 1])


        """
        Complete the triangle consisting of: 
            1.  the vector from the center of the mirror to the projector's
                focal point (completely specified)
            2.  the vector from the projector's focal point to the mirror for
                the given projector pixel (known direction, unknown length)
            3.  the vector from the center of the mirror to the point on the
                mirror where the vector in 2 hits the mirror (known length,
                unknown direction)
        Vector 3 is normal to the mirror's surface at the point of reflection
        and is used to find the direction of the reflected light.
        """
        # solve quadratic equation for y-component of vector 2
        px = self._projectorPosition[0]
        py = self._projectorPosition[1]
        pz = self._projectorPosition[2]
        pdx = self._projectorPixelDirections[:, :, 0]
        pdy = self._projectorPixelDirections[:, :, 1]
        pdz = self._projectorPixelDirections[:, :, 2]
        a0 = px**2 + py**2 + pz**2 - mirrorRadius**2
        a1 = 2*px*pdx + 2*py*pdy + 2*pz*pdz
        a2 = pdx**2 + pdy**2 + pdz**2
        projectorMask = zeros([projectorPixelHeight,
                         projectorPixelWidth], dtype=int)
        incidentLightVectors = zeros([projectorPixelHeight,
                         projectorPixelWidth, 3])
        for i in range(projectorPixelHeight):
            for j in range(projectorPixelWidth):
                """
                The vector will intersect the sphere twice.  Find the root
                for the shorter vector.
                """
                r = min(roots(array([a2[i, j], a1[i, j], a0])))
                if imag(r) == 0:
                    """
                    For projector pixels that hit the mirror, calculate the
                    incident light vector and set the mask to one.
                    """
                    x = r*pdx[i, j]
                    y = r*pdy[i, j]
                    z = r*pdz[i, j]
                    incidentLightVectors[i, j] = array([x, y, z])
                    projectorMask[i, j] = 1

        mirrorRadiusVectors = self._projectorPosition + incidentLightVectors
        mirrorUnitNormals = mirrorRadiusVectors / mirrorRadius

        if DEBUG:
            self._projectorMask = projectorMask
            self._incidentLightVectors = incidentLightVectors
            self._mirrorRadiusVectors = mirrorRadiusVectors
            self._mirrorUnitNormals = mirrorUnitNormals

        """
        Use the incidentLightVectors and the mirrorUnitNormals to find the
        direction of the reflected light.
        """
        reflectedLightDirections = zeros([projectorPixelHeight,
                         projectorPixelWidth, 3])
        for i in range(projectorPixelHeight):
            for j in range(projectorPixelWidth):
                if projectorMask[i, j] == 1:
                    u = mirrorUnitNormals[i, j]
                    reflectedLightVector = \
                        -2*dot(incidentLightVectors[i, j], u)*u \
                        + incidentLightVectors[i, j]
                    reflectedLightDirections[i, j] = \
                        reflectedLightVector/linalg.norm(reflectedLightVector)

        if DEBUG:
            self._reflectedLightDirections = reflectedLightDirections

        """
        Complete the triangle again to find the reflected light vectors.
        The known vector is from the center of the dome to the reflection
        point on the mirror (calculated as mirrorRadiusVectors - domeCenter)
        and the length of the vector with unknown direction is the dome radius.
        """
        # solve quadratic for y-component of reflected light vectors
        rpx = mirrorRadiusVectors[:, :, 0] - domeCenter[0]
        rpy = mirrorRadiusVectors[:, :, 1] - domeCenter[1]
        rpz = mirrorRadiusVectors[:, :, 2] - domeCenter[2]
        rldx = reflectedLightDirections[:, :, 0]
        rldy = reflectedLightDirections[:, :, 1]
        rldz = reflectedLightDirections[:, :, 2]
        a0 = rpx**2 + rpy**2 + rpz**2 - domeRadius**2
        a1 = 2*rpx*rldx + 2*rpy*rldy + 2*rpz*rldz
        a2 = rldx**2 + rldy**2 + rldz**2
        reflectedLightVectors = zeros([projectorPixelHeight,
                         projectorPixelWidth, 3])
        for i in range(projectorPixelHeight):
            for j in range(projectorPixelWidth):
                if projectorMask[i, j] == 1:
                    # For projector pixels that hit the mirror,
                    # take the solution with positive vector length.
                    r = max(roots(array([a2[i, j], a1[i, j], a0[i, j]])))
                    x = r*rldx[i, j]
                    y = r*rldy[i, j]
                    z = r*rldz[i, j]
                    reflectedLightVectors[i, j] = [x, y, z]

        if DEBUG:
            self._reflectedLightVectors = reflectedLightVectors

        """
        Now use the vectors of the reflected light, reflection position on the
        mirror, and mouse position to find the mouse's view point
        """
        mouseViewDirections = zeros([projectorPixelHeight,
                                     projectorPixelWidth, 3])
        for i in range(projectorPixelHeight):
            for j in range(projectorPixelWidth):
                if projectorMask[i, j] == 1:
                    # For projector pixels that hit the mirror,
                    # calculate the view direction for the mouse.
                    mouseViewVector = (reflectedLightVectors[i, j]
                                       + mirrorRadiusVectors[i, j]
                                       - mousePosition)
                    magnitude = linalg.norm(mouseViewVector)
                    mouseViewDirections[i, j] = mouseViewVector/magnitude

        self._mouseViewDirections = mouseViewDirections
        
        """
        For each OpenGL image pixel use the directions calculated above
        to find the projector pixel with the closest direction.  Then add that
        OpenGL pixel to the projector pixel's list of contributing pixels.
        """

        # This 2D list of lists contains the list of OpenGL pixels
        # that contribute to each projector pixel.
        self._contributingPixels = \
            [[[] for i in range(self._projectorPixelWidth)]
             for j in range(self._projectorPixelHeight)]
        row = 0
        while row < self._imagePixelHeight:
            for col in range(self._imagePixelWidth):
                """
                For each OpenGL image pixel, determine which projector
                pixel has the closest direction.  
                """
                [r, c] = self._find_closest_projector_pixel(row, col) 
                self._contributingPixels[r][c].append([row, col])
            row = row + 1
            for col in range(self._imagePixelWidth - 1, -1, -1):
                """
                Go through the pixels in a serpentine pattern so that the
                current pixel is always close to the last pixel.  This way the
                search algorithm can use the last result as its starting point.
                """
                [r, c] = self._find_closest_projector_pixel(row, col) 
                self._contributingPixels[r][c].append([row, col])
            row = row + 1


    #####################################################################
    # Class methods
    #####################################################################

    def _find_projector_focal_point(self):
        """
        Find the position of the projector's focal point.  The projector
        image is horizontally centered on the mirror so the x-component
        of the focal point's position is zero.  Find the intersection point
        of the lines along the top and bottom of the projected light to get
        the focal point's y and z coordinates.
        """
        # calculate slope of line along top of projected light
        upper_z1 = self._firstProjectorImage[0][2]
        upper_z2 = self._secondProjectorImage[0][2]
        y1 = self._firstProjectorImage[0][1]
        y2 = self._secondProjectorImage[0][1]
        upperSlope = (upper_z2 - upper_z1)/(y2 - y1)

        # calculate slope of line along bottom of projected light
        lower_z1 = self._firstProjectorImage[2][2]
        lower_z2 = self._secondProjectorImage[2][2]
        lowerSlope = (lower_z2 - lower_z1)/(y2 - y1)
        
        # find y and z where the lines intersect
        a = array([[upperSlope, -1], [lowerSlope, -1]])
        b = array([upperSlope*y1 - upper_z1, lowerSlope*y1 - lower_z1])
        [y, z] = linalg.solve(a, b)
        projectorPosition = array([0, y, z])

        return projectorPosition


    def _find_closest_projector_pixel(self, row, col):
        """
        For the OpenGL image pixel specified by row and col use the directions
        in self._cameraViewDirections and self._mouseViewDirections to find the
        projector pixel which has the closest direction and return its row and
        column in a list.  This is done using a search method rather than
        calculating the dot product for every projector pixel.
        """
        cameraDirection = self._cameraViewDirections[row, col]

        # Start with the last projector pixel
        r = self._projectorPixelRow
        c = self._projectorPixelCol
        mouseDirection = self._mouseViewDirections[r, c]

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
            neighborDirection = self._mouseViewDirections[neighbor[0], neighbor[1]]
            neighbor_dps.append(dot(cameraDirection, neighborDirection))

        return [neighbor_dps, neighbors]


    def _showGeometry(self, row, col):
        """
        Print the vectors associated with the projector pixel specified by
        row and column.
        """

        print "self._projectorPixelDirections[row, col]"
        print self._projectorPixelDirections[row, col]
        print "self._incidentLightVectors[row, col]"
        print self._incidentLightVectors[row, col]
        print "self._mirrorRadiusVectors[row, col]"
        print self._mirrorRadiusVectors[row, col]
        print "self._reflectedLightVectors[row, col]"
        print self._reflectedLightVectors[row, col]
        print "self._mouseViewDirections[row, col]"
        print self._mouseViewDirections[row, col]
        import pylab
        # plot the geometry
        pylab.title('Geometry for projector pixel:' + str(row) + "," + str(col))
        projectorPositionVector = [[0, self._projectorPosition[1]], [0, self._projectorPosition[2]]]
        pylab.plot(projectorPositionVector)
        pylab.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1)



    def warpImageForDome(self, image):
        """
        Take an RGB input image intended for display on a flat screen and
        produce an image for the projector that removes the distortions caused
        by projecting the image onto the dome using a spherical mirror.
        """
        assert image.size == (self._imagePixelWidth, self._imagePixelHeight)

        pixels = array(image)
        warpedPixels = zeros([self._projectorPixelHeight,
                              self._projectorPixelWidth, 3], dtype=uint8)
        for row in range(self._projectorPixelHeight):
            for col in range(self._projectorPixelWidth):
                pixelValue = zeros(3)
                for pixel in self._contributingPixels[row][col]:
                    pixelValue += pixels[pixel[0], pixel[1]]
                n = len(self._contributingPixels[row][col])
                if n > 0:
                    pixelValue = pixelValue/n
                warpedPixels[row][col] = array(pixelValue, dtype=uint8)

        return Image.fromarray(warpedPixels, mode='RGB')



#####################################################################
# Functions called by class methods
#####################################################################

def flatDisplayDirections(screenHeight, screenWidth, pixelHeight, pixelWidth,
                          distanceToScreen, verticalOffset = 0):
    """
    Return unit vectors that point from the viewer towards each pixel
    on a flat screen display.  The display is along the positive y-axis
    relative to the viewer.  The positive x-axis is to the viewer's right
    and the positive z-axis is up.
    """
    # Make matrices of projector row and column values
    rows = array([[float(i)]*pixelWidth for i in 
                  range(pixelHeight)])
    columns = array([[float(i)]*pixelHeight for i in
                     range(pixelWidth)]).T

    """
    Calculate x and z values from column and row values so they
    are symmetric about the center of the image and scaled to the
    screen size.
    """
    x = screenWidth*(columns/(pixelWidth - 1) - 0.5)
    z = -screenHeight*(rows/(pixelHeight - 1) - 0.5) + verticalOffset

    # y is the distance from the viewer to the screen
    y = distanceToScreen
    r = sqrt(x**2 + y**2 + z**2)

    return dstack([x/r, y/r, z/r])

    

def plotMagnitude(arrayOfVectors):
    """
    Plot an image of the magnitude of 3D vectors which are stored in a 2D
    array.  The luminance of each pixel is proportional to the normalized
    magnitude of its vector so larger magnitudes have lighter pixels.
    """
    dimensions = shape(arrayOfVectors)
    #pixels = ones(dimensions, dtype=uint8)
    magnitudes = linalg.norm(arrayOfVectors, axis=-1)
    normalizationFactor = max(magnitudes.flat)
    pixels = array(255*magnitudes/normalizationFactor, dtype=uint8)
    magnitudeImage = Image.fromarray(pixels, mode='L')
    magnitudeImage.show()



############################################################
# Main Program Starts Here
############################################################
from numpy import *
from PIL import Image

if __name__ == "__main__":
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


