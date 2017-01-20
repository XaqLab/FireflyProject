from numpy import array
import sys, os

from dome_figure import DomeFigure


# Create the figure and pick colors for each object.
figure = DomeFigure(size=(1560,600))
projector = figure.colors['light_gray']
light = figure.colors['red']
frustum = figure.colors['blue']
annotations = figure.colors['green']
# Specify line properties.
dashed = (0.005, 0.005)
solid = None
line_width = 3
cone_size = 0.03
# Specify some vectors.
distance_to_image = array([0,-0.5,0])
image_width = array([0.16,0,0])
image_height = array([0,0,0.09])
image_rise = array([0,0,0.05])
image_center = (figure.focal_point + distance_to_image + image_rise)
upper_left_corner = (figure.focal_point + distance_to_image
                     - 0.5*image_width + 0.5*image_height + image_rise)
upper_right_corner = (figure.focal_point + distance_to_image
                      + 0.5*image_width + 0.5*image_height + image_rise)
lower_left_corner = (figure.focal_point + distance_to_image
                     - 0.5*image_width - 0.5*image_height + image_rise)
lower_right_corner = (figure.focal_point + distance_to_image
                      + 0.5*image_width - 0.5*image_height + image_rise)
left_midpoint = (figure.focal_point + distance_to_image
                 - 0.5*image_width + image_rise)
right_midpoint = (figure.focal_point + distance_to_image
                  + 0.5*image_width + image_rise)
point_in_image = (figure.focal_point + distance_to_image
                  + 0.33*image_width - 0.13*image_height + image_rise)

""" Draw the projector. """
# Draw a 3D extrusion for the projector.
figure.draw_projector(color=projector, opacity=0.25)

""" Draw the outline of the projector's frustum. """
# Draw a rectange for the projector's image.
figure.draw_line(upper_left_corner, upper_right_corner,
                 line_width=line_width, color=frustum, opacity=1.0)
figure.draw_line(upper_right_corner, lower_right_corner,
                 line_width=line_width, color=frustum, opacity=1.0)
figure.draw_line(lower_right_corner, lower_left_corner,
                 line_width=line_width, color=frustum, opacity=1.0)
figure.draw_line(lower_left_corner, upper_left_corner,
                 line_width=line_width, color=frustum, opacity=1.0)
# Draw the lines from the projector's focal point to the image corners.
figure.draw_line(figure.focal_point, upper_left_corner,
                 line_width=line_width, color=frustum, opacity=1.0,
                 pattern=dashed)
figure.draw_line(figure.focal_point, upper_right_corner,
                 line_width=line_width, color=frustum, opacity=1.0,
                 pattern=dashed)
figure.draw_line(figure.focal_point, lower_left_corner,
                 line_width=line_width, color=frustum, opacity=1.0,
                 pattern=dashed)
figure.draw_line(figure.focal_point, lower_right_corner,
                 line_width=line_width, color=frustum, opacity=1.0,
                 pattern=dashed)

""" Draw the light from the projector's focal point to a single point in the
projector's image. """
# draw a line from the projector's focal point to the point on the mirror
figure.draw_line(figure.focal_point, point_in_image,
                 line_width=line_width, color=light, opacity=1.0)
figure.draw_cone(point_in_image, point_in_image - figure.focal_point,
                 cone_size, color=light)

""" Draw the annotations for theta and vertical_offset. """
figure.draw_line(figure.focal_point, lower_left_corner - image_rise,
                 line_width=line_width, color=annotations, opacity=1.0,
                 pattern=dashed)
figure.draw_line(figure.focal_point, lower_right_corner - image_rise,
                 line_width=line_width, color=annotations, opacity=1.0,
                 pattern=dashed)
figure.draw_line(lower_left_corner - image_rise,
                 lower_right_corner - image_rise,
                 line_width=line_width, color=annotations, opacity=1.0,
                 pattern=dashed)
figure.draw_line(lower_left_corner - image_rise, lower_left_corner,
                 line_width=line_width, color=annotations, opacity=1.0,
                 pattern=dashed)
figure.draw_line(lower_right_corner - image_rise, lower_right_corner,
                 line_width=line_width, color=annotations, opacity=1.0,
                 pattern=dashed)

# orient the figure before saving 
# side view
#figure.mlab.view(-10.0, 92.0, 1.7, array([0.001, 0.475, 0.4]))

# upper left view
#figure.mlab.view(-46.8, 70.0, 1.42, array([0, 0.725, 0.0825]))
#figure.mlab.view(-50.0, 85.0, 1.41, array([0, 0.725, 0.0725]))
#figure.mlab.view(-50.0, 85.0, 1.41, array([-0.32, 1.11, 0.0289]))
figure.mlab.view(-50.0, 85.0, 1.41, array([-0.556,  1.236, -0.023]))
#figure.figure.scene.camera.zoom(1.8)
#figure.mlab.move(right=-0.03)
#import ipdb; ipdb.set_trace()


filename = os.path.splitext(sys.argv[0])[0]
figure.save(filename + ".tiff", magnification=1)


