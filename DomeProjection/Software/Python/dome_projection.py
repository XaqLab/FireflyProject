#!/usr/local/bin/python
# written for python 2.7.8

"""
This is the mapping to project images from OpenGL onto the hemispherical dome
display for animal virtual reality experiments.  The objective of this mapping
is to ensure that the OpenGL image and the image on the dome display look the
same from the viewer's point of view.  For the dome display the viewer is the
animal and for the OpenGL image the viewer is a virtual camera.  Each projector
pixel projects to a small region seen by the animal.  The RGB values for each
projector pixel are calculated from pixels in the OpenGL image such that the
pixels in the OpenGL image and the corresponding region seen by the animal are
in the same direction.  A mapping from projector pixels to the animal's
viewpoint is used to determine which pixels in the OpenGL image contribute to
each projector pixel.  The RGB values for each projector pixel are the 
average of the RGB values from the OpenGL image contributing pixels.
"""

DEBUG = True

class DomeDisplay:
    """
    The dome display class describes the geometry of our dome, spherical
    mirror, and projector along with the geometry of the associated OpenGL
    virtual camera and screen.
    All vectors in the OpenGL world are relative to the virtual camera and
    all vectors in the dome display setup are relative to the center of the
    hemispherical mirror.
    """
    def __init__(self,
                 screen_height = 1,
                 screen_width = 1,
                 distance_to_screen = 2.0,
                 image_pixel_height = 512,
                 image_pixel_width = 512,
                 projector_pixel_height = 720,
                 projector_pixel_width = 1280,
                 first_projector_image = [[-0.09, 0.43, 0.18],
                                          [0.09, 0.43, 0.18],
                                          [0.09, 0.43, 0.04],
                                          [-0.09, 0.43, 0.04]],
                 second_projector_image = [[-0.11, 0.36, 0.21],
                                           [0.11, 0.36, 0.21],
                                           [0.11, 0.36, 0.04],
                                           [-0.11, 0.36, 0.04]],
                 #mirror_radius = 0.2286,
                 mirror_radius = 0.3,
                 dome_center = [0, 0.14, 0.42],
                 dome_radius = 0.64,
                 animal_position = [0, 0, 0.5]
                 ):

        """
        Parameters:
        ----------------------------
        @param screen_height: 
            The height of the OpenGL screen in arbitrary units.
        @param screen_width: 
            The width of the OpenGL screen in arbitrary units.
        @param distance_to_screen: 
            The distance from OpenGL's virtual camera to the OpenGL screen in
            arbitrary units.
        @param image_pixel_height:
            The number of vertical pixels in the OpenGL image.
        @param image_pixel_width:
            The number of horizontal pixels in the OpenGL image.
        @param projector_pixel_height:
            The number of vertical pixels in the projector image.
        @param projector_pixel_width:
            The number of horizontal pixels in the projector image.
        @param first_projector_image:
            A list of four (x,y1,z) points, starting top left and proceeding
            clockwise, that specifies the corners of the projector's image
            at a distance y1 from the center of the mirror.
        @param second_projector_image:
            A list of four (x,y2,z) points, starting top left and proceeding
            clockwise, that specifies the corners of the projector's image
            at a distance y2 from the center of the mirror.
        @param mirror_radius:
            The radius of the mirror in arbitrary units.
        @param dome_center:
            An (x,y,z) vector from the center of the mirror to the center of
            the dome.
        @param dome_radius:
            The radius of the dome in arbitrary units.
        @param animal_position:
            An (x,y,z) vector from the center of the mirror to the position
            of the animal's eyes.
        """

        #######################################################################
        # Properties passed in as arguments
        #######################################################################
        self._screen_height = screen_height
        self._screen_width = screen_width
        self._distance_to_screen = distance_to_screen
        self._image_pixel_height = image_pixel_height
        self._image_pixel_width = image_pixel_width
        self._projector_pixel_height = projector_pixel_height
        self._projector_pixel_width = projector_pixel_width
        self._first_projector_image = first_projector_image
        self._second_projector_image = second_projector_image
        self._mirror_radius = mirror_radius
        self._dome_center = dome_center
        self._dome_radius = dome_radius
        self._animal_position = animal_position

        #######################################################################
        # Properties used to share results between method calls
        #######################################################################

        # Start search low in the middle of the projector image 
        self._projector_pixel_row = 3*projector_pixel_height/4
        self._projector_pixel_col = projector_pixel_width/2

        #######################################################################
        # Properties calculated from arguments
        #######################################################################

        """
        Calculate the unit vectors (aka directions) that point from
        OpenGL's virtual camera towards all of the OpenGL image pixels.
        All vectors are relative to OpenGL's virtual camera which
        is looking down the positive y-axis.
        """
        self._camera_view_directions = \
            flat_display_directions(screen_height, screen_width,
                                    image_pixel_height, image_pixel_width,
                                    distance_to_screen)


        """
        Calculate the unit vectors (directions) from the animal inside the dome
        towards the projection of each projector pixel on to the dome.
        """
        self._animal_view_directions = self._dome_display_directions()


        """
        For each OpenGL image pixel use the directions calculated above
        to find the projector pixel with the closest direction.  Then add that
        OpenGL pixel to the projector pixel's list of contributing pixels.
        """

        # This 2D list of lists contains the list of OpenGL pixels
        # that contribute to each projector pixel.
        self._contributing_pixels = \
            [[[] for i in range(self._projector_pixel_width)]
             for j in range(self._projector_pixel_height)]
        row = 0
        while row < self._image_pixel_height:
            for col in range(self._image_pixel_width):
                """
                For each OpenGL image pixel, determine which projector
                pixel has the closest direction.  
                """
                [r, c] = self._find_closest_projector_pixel(row, col) 
                self._contributing_pixels[r][c].append([row, col])
            row = row + 1
            for col in range(self._image_pixel_width - 1, -1, -1):
                """
                Go through the pixels in a serpentine pattern so that the
                current pixel is always close to the last pixel.  This way the
                search algorithm can use the last result as its starting point.
                """
                [r, c] = self._find_closest_projector_pixel(row, col) 
                self._contributing_pixels[r][c].append([row, col])
            row = row + 1


    ##########################################################################
    # Class methods
    ##########################################################################

    def _find_projector_focal_point(self):
        """
        Find the position of the projector's focal point.  The projector
        image is horizontally centered on the mirror so the x-component
        of the focal point's position is zero.  Find the intersection point
        of the lines along the top and bottom of the projected light to get
        the focal point's y and z coordinates.
        """
        # calculate slope of line along top of projected light
        upper_z1 = self._first_projector_image[0][2]
        upper_z2 = self._second_projector_image[0][2]
        y1 = self._first_projector_image[0][1]
        y2 = self._second_projector_image[0][1]
        upperSlope = (upper_z2 - upper_z1)/(y2 - y1)

        # calculate slope of line along bottom of projected light
        lower_z1 = self._first_projector_image[2][2]
        lower_z2 = self._second_projector_image[2][2]
        lowerSlope = (lower_z2 - lower_z1)/(y2 - y1)
        
        # find y and z where the lines intersect
        a = array([[upperSlope, -1], [lowerSlope, -1]])
        b = array([upperSlope*y1 - upper_z1, lowerSlope*y1 - lower_z1])
        [y, z] = linalg.solve(a, b)
        projector_focal_point = array([0, y, z])

        return projector_focal_point


    def _dome_display_directions(self):
        """
        Return the unit vectors (directions) from the viewer inside the dome
        towards the projection of each projector pixel on to the dome.
        All vectors used in these calculations are relative to the center of
        the hemispherical mirror.  The projector is on the positive y-axis
        (but projecting in the -y direction) and its projected image is assumed
        to be horizontally centered on the mirror.
        """


        # Find the position of the projector's focal point.
        projector_focal_point = self._find_projector_focal_point()

        if DEBUG:
            self._projector_focal_point = projector_focal_point


        """
        Calculate the unit vectors (directions) from the projector's focal
        point towards the mirror for each projector pixel.  This calculation
        assumes second_projector_image has the same width at the top and bottom.
        """
        # calculate image height and width
        top_right_x = self._second_projector_image[1][0]
        top_left_x = self._second_projector_image[0][0]
        top_left_z = self._second_projector_image[0][2]
        bottom_left_z = self._second_projector_image[3][2]

        image_height = top_left_z - bottom_left_z
        image_width = top_right_x - top_left_x

        # calculate distance from projector to image
        image_y = self._second_projector_image[0][1]
        projector_y = projector_focal_point[1]

        distance_to_image = image_y - projector_y

        # calculate vertical offset of image relative to projector
        projector_z = projector_focal_point[2]

        vertical_offset = bottom_left_z - projector_z + image_height/2

        # calculate the directions for each projector pixel
        self._projector_pixel_directions = \
            flat_display_directions(image_height,
                                    image_width,
                                    self._projector_pixel_height,
                                    self._projector_pixel_width,
                                    distance_to_image,
                                    vertical_offset)

        # Flip the sign of the x-values because projection is in -y direction
        self._projector_pixel_directions *= array([-1, 1, 1])


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
        px = projector_focal_point[0]
        py = projector_focal_point[1]
        pz = projector_focal_point[2]
        pdx = self._projector_pixel_directions[:, :, 0]
        pdy = self._projector_pixel_directions[:, :, 1]
        pdz = self._projector_pixel_directions[:, :, 2]
        a0 = px**2 + py**2 + pz**2 - self._mirror_radius**2
        a1 = 2*px*pdx + 2*py*pdy + 2*pz*pdz
        a2 = pdx**2 + pdy**2 + pdz**2
        projector_mask = zeros([self._projector_pixel_height,
                                self._projector_pixel_width], dtype=int)
        incident_light_vectors = zeros([self._projector_pixel_height,
                                        self._projector_pixel_width, 3])
        for i in range(self._projector_pixel_height):
            for j in range(self._projector_pixel_width):
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
                    incident_light_vectors[i, j] = array([x, y, z])
                    projector_mask[i, j] = 1
    
        mirror_radius_vectors = projector_focal_point + incident_light_vectors
        mirrorUnitNormals = mirror_radius_vectors / self._mirror_radius
    
        if DEBUG:
            # create properties for intermediate results
            self._projector_mask = projector_mask
            self._incident_light_vectors = incident_light_vectors
            self._mirror_radius_vectors = mirror_radius_vectors
            self._mirrorUnitNormals = mirrorUnitNormals
    
        """
        Use the incident_light_vectors and the mirrorUnitNormals to find the
        direction of the reflected light.
        """
        reflectedLightDirections = zeros([self._projector_pixel_height,
                                          self._projector_pixel_width, 3])
        for i in range(self._projector_pixel_height):
            for j in range(self._projector_pixel_width):
                if projector_mask[i, j] == 1:
                    u = mirrorUnitNormals[i, j]
                    reflectedLightVector = \
                        -2*dot(incident_light_vectors[i, j], u)*u \
                        + incident_light_vectors[i, j]
                    reflectedLightDirections[i, j] = \
                        reflectedLightVector/linalg.norm(reflectedLightVector)
    
        if DEBUG:
            self._reflectedLightDirections = reflectedLightDirections
    
        """
        Complete the triangle again to find the reflected light vectors.
        The known vector is from the center of the dome to the reflection
        point on the mirror (calculated as mirror_radius_vectors - dome_center)
        and the length of the vector with unknown direction is the dome radius.
        """
        # solve quadratic for y-component of reflected light vectors
        rpx = mirror_radius_vectors[:, :, 0] - self._dome_center[0]
        rpy = mirror_radius_vectors[:, :, 1] - self._dome_center[1]
        rpz = mirror_radius_vectors[:, :, 2] - self._dome_center[2]
        rldx = reflectedLightDirections[:, :, 0]
        rldy = reflectedLightDirections[:, :, 1]
        rldz = reflectedLightDirections[:, :, 2]
        a0 = rpx**2 + rpy**2 + rpz**2 - self._dome_radius**2
        a1 = 2*rpx*rldx + 2*rpy*rldy + 2*rpz*rldz
        a2 = rldx**2 + rldy**2 + rldz**2
        reflected_light_vectors = zeros([self._projector_pixel_height,
                                         self._projector_pixel_width, 3])
        for i in range(self._projector_pixel_height):
            for j in range(self._projector_pixel_width):
                if projector_mask[i, j] == 1:
                    # For projector pixels that hit the mirror,
                    # take the solution with positive vector length.
                    r = max(roots(array([a2[i, j], a1[i, j], a0[i, j]])))
                    x = r*rldx[i, j]
                    y = r*rldy[i, j]
                    z = r*rldz[i, j]
                    reflected_light_vectors[i, j] = [x, y, z]
    
        if DEBUG:
            self._reflected_light_vectors = reflected_light_vectors
    
        """
        Now use the vectors of the reflected light, reflection position on the
        mirror, and animal position to find the animal's view point
        """
        animal_view_directions = zeros([self._projector_pixel_height,
                                        self._projector_pixel_width, 3])
        for i in range(self._projector_pixel_height):
            for j in range(self._projector_pixel_width):
                if projector_mask[i, j] == 1:
                    # For projector pixels that hit the mirror,
                    # calculate the view direction for the animal.
                    animal_view_vector = (reflected_light_vectors[i, j]
                                        + mirror_radius_vectors[i, j]
                                        - self._animal_position)
                    magnitude = linalg.norm(animal_view_vector)
                    animal_view_directions[i, j] = animal_view_vector/magnitude
    
        return animal_view_directions
        

    def _find_closest_projector_pixel(self, row, col):
        """
        For the OpenGL image pixel specified by row and col use the directions
        in self._camera_view_directions and self._animal_view_directions to find the
        projector pixel which has the closest direction and return its row and
        column in a list.  This is done using a search method rather than
        calculating the dot product for every projector pixel.
        """
        camera_direction = self._camera_view_directions[row, col]

        # Start with the last projector pixel
        r = self._projector_pixel_row
        c = self._projector_pixel_col
        animalDirection = self._animal_view_directions[r, c]

        # Calculate dot product of this OpenGL pixel
        # with the last projector pixel.
        dp = dot(camera_direction, animalDirection)

        # Calculate dot product of this OpenGL pixel with the
        # neighboring projector pixels.
        [neighbor_dps, neighbors] = \
            self._calc_neighbor_dot_products(r, c, camera_direction)

        while max(neighbor_dps) > dp:
            """
            If the dot product with one of the neighboring projector pixels is 
            larger then update r and c to that pixel and check its neighbors.
            Repeat until all neighbors have smaller (or equal) dot products.
            """
            dp = max(neighbor_dps)
            [r, c] = neighbors[neighbor_dps.index(dp)]
            [neighbor_dps, neighbors] = \
                self._calc_neighbor_dot_products(r, c, camera_direction)

        # Save projector pixel for next call
        self._projector_pixel_row = r
        self._projector_pixel_col = c
        return [r, c]


    def _calc_neighbor_dot_products(self, row, col, camera_direction):
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
            if row + dr >= 0 and row + dr < self._projector_pixel_height:
                if col + dc >= 0 and col + dc < self._projector_pixel_width:
                    neighbors.append([row + dr, col + dc])

        # calculate neighbor dot products
        for neighbor in neighbors:
            neighbor_direction = \
                self._animal_view_directions[neighbor[0], neighbor[1]]
            neighbor_dps.append(dot(camera_direction, neighbor_direction))

        return [neighbor_dps, neighbors]


    def _debug_geometry(self, row, col):
        """
        Display images of intermediate results.
        """

        x_unit_vector = array([1, 0, 0])
        y_unit_vector = array([0, 1, 0])
        z_unit_vector = array([0, 0, 1])

        projector_y = self._projector_pixel_directions.dot(y_unit_vector)
        image = Image.fromarray(array(255*projector_y, dtype=uint8), mode='L')
        image.show()

        image = Image.fromarray(array(255*self._projector_mask,
                                     dtype=uint8), mode='L')
        image.show()

        animal_view_y = self._animal_view_directions.dot(y_unit_vector)
        y_image = Image.fromarray(array(255*abs(animal_view_y),
                                        dtype=uint8), mode='L')
        y_image.show()


    def _show_geometry(self, row, col):
        """
        Print the vectors associated with the projector pixel specified by
        row and column.
        """

        print "self._projector_pixel_directions[row, col]"
        print self._projector_pixel_directions[row, col]
        print "self._incident_light_vectors[row, col]"
        print self._incident_light_vectors[row, col]
        print "self._mirror_radius_vectors[row, col]"
        print self._mirror_radius_vectors[row, col]
        print "self._reflected_light_vectors[row, col]"
        print self._reflected_light_vectors[row, col]
        print "self._animal_view_directions[row, col]"
        print self._animal_view_directions[row, col]
        import pylab
        # plot the geometry
        pylab.title('Geometry for projector pixel:' + str(row) + "," + str(col))
        projector_focal_point_vector = \
            [[0, self._projector_focal_point[1]],
             [0, self._projector_focal_point[2]]]
        pylab.plot(projector_focal_point_vector)
        pylab.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1)


    def warp_image_for_dome(self, image):
        """
        Take an RGB input image intended for display on a flat screen and
        produce an image for the projector that removes the distortions caused
        by projecting the image onto the dome using a spherical mirror.
        """
        assert image.size == (self._image_pixel_width,
                              self._image_pixel_height)

        pixels = array(image)
        warped_pixels = zeros([self._projector_pixel_height,
                              self._projector_pixel_width, 3], dtype=uint8)
        for row in range(self._projector_pixel_height):
            for col in range(self._projector_pixel_width):
                pixel_value = zeros(3)
                for pixel in self._contributing_pixels[row][col]:
                    pixel_value += pixels[pixel[0], pixel[1]]
                n = len(self._contributing_pixels[row][col])
                if n > 0:
                    pixel_value = pixel_value/n
                warped_pixels[row][col] = array(pixel_value, dtype=uint8)

        return Image.fromarray(warped_pixels, mode='RGB')



###############################################################################
# Functions called by class methods
###############################################################################

def flat_display_directions(screen_height, screen_width, pixel_height,
                            pixel_width, distance_to_screen,
                            vertical_offset = 0):
    """
    Return unit vectors that point from the viewer towards each pixel
    on a flat screen display.  The display is along the positive y-axis
    relative to the viewer.  The positive x-axis is to the viewer's right
    and the positive z-axis is up.
    """
    # Make matrices of projector row and column values
    rows = array([[float(i)]*pixel_width for i in 
                  range(pixel_height)])
    columns = array([[float(i)]*pixel_height for i in
                     range(pixel_width)]).T

    """
    Calculate x and z values from column and row values so they
    are symmetric about the center of the image and scaled to the
    screen size.
    """
    x = screen_width*(columns/(pixel_width - 1) - 0.5)
    z = -screen_height*(rows/(pixel_height - 1) - 0.5) + vertical_offset

    # y is the distance from the viewer to the screen
    y = distance_to_screen
    r = sqrt(x**2 + y**2 + z**2)

    return dstack([x/r, y/r, z/r])

    
def plot_magnitude(array_of_vectors):
    """
    Plot an image of the magnitude of 3D vectors which are stored in a 2D
    array.  The luminance of each pixel is proportional to the normalized
    magnitude of its vector so larger magnitudes have lighter pixels.
    """
    dimensions = shape(array_of_vectors)
    #pixels = ones(dimensions, dtype=uint8)
    magnitudes = linalg.norm(array_of_vectors, axis=-1)
    normalizationFactor = max(magnitudes.flat)
    pixels = array(255*magnitudes/normalizationFactor, dtype=uint8)
    magnitude_image = Image.fromarray(pixels, mode='L')
    magnitude_image.show()



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
    dome = DomeDisplay()
    print "Done initializing dome"
    
    #output_image = Image.fromarray(pixels, mode='RGB')
    output_image = dome.warp_image_for_dome(input_image)
    
    output_image.show()
    #output_image.save("output_file", "PNG")


