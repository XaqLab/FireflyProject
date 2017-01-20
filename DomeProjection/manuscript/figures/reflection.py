from numpy import array
from numpy.linalg import norm
from mayavi import mlab
import sys, os

from dome_figure import DomeFigure


# Create the figure and pick colors for each object.
figure = DomeFigure(size=(1560,830))
mirror = figure.colors['silver']
light = figure.colors['red']
normal = figure.colors['black']
# Specify line properties.
dashed = (0.01, 0.01)
line_width = 3
cone_size = 0.03
# Specify a vector representing the direction of the incident light and make
# the size of the vector the radius of the mirror.
direction = array([0.0, -0.2, 0.02])
incident_vector = figure.mirror_radius*direction/norm(direction)

""" Draw the mirror. """
figure.draw_mirror(color=mirror, opacity=0.25)

""" Draw vectors representing the incident light, the direction normal to the
surface of the mirror and the reflected light. """
# Draw a vector for the incident light.
figure.draw_line(figure.point_on_mirror,
                 figure.point_on_mirror - incident_vector,
                 line_width=line_width, color=light, opacity=1.0)
figure.draw_cone(figure.point_on_mirror, incident_vector, cone_size,
                 color=light)
# Draw a line perpendicular to the mirror's surface.
figure.draw_line(figure.mirror_center, figure.point_on_mirror,
                 line_width=line_width, color=normal, opacity=1.0)
figure.draw_cone(figure.point_on_mirror, figure.point_on_mirror, cone_size,
                 color=normal)
# Calculate reflected light vector.
unit_normal = figure.point_on_mirror/norm(figure.point_on_mirror)
direction = incident_vector - 2*incident_vector.dot(unit_normal)*unit_normal
reflected_vector = figure.mirror_radius*direction/norm(direction)
# Draw a line for the reflected light.
figure.draw_line(figure.point_on_mirror,
                 figure.point_on_mirror + reflected_vector,
                 line_width=line_width, color=light, opacity=1.0)
figure.draw_cone(figure.point_on_mirror + reflected_vector, reflected_vector,
                 cone_size, color=light)

""" Draw vectors illustrating calculation of the reflected light direction. """
# Draw the projection of incident light onto the surface normal.
figure.draw_line(figure.point_on_mirror,
                 figure.point_on_mirror -
                 incident_vector.dot(unit_normal)*unit_normal,
                 line_width=line_width, color=normal, opacity=1.0,
                 pattern=dashed)
figure.draw_cone(figure.point_on_mirror -
                 incident_vector.dot(unit_normal)*unit_normal,
                 unit_normal, cone_size, color=normal)
# Extend the projection of incident light onto the surface normal.
figure.draw_line(figure.point_on_mirror -
                 incident_vector.dot(unit_normal)*unit_normal,
                 figure.point_on_mirror -
                 2*incident_vector.dot(unit_normal)*unit_normal,
                 line_width=line_width, color=normal, opacity=1.0,
                 pattern=dashed)
figure.draw_cone(figure.point_on_mirror -
                 2*incident_vector.dot(unit_normal)*unit_normal,
                 unit_normal, cone_size, color=normal)
# Draw -1*reflected light starting at the end of the extended projection.
figure.draw_line(figure.point_on_mirror -
                 2*incident_vector.dot(unit_normal)*unit_normal,
                 figure.point_on_mirror -
                 2*incident_vector.dot(unit_normal)*unit_normal -
                 reflected_vector,
                 line_width=line_width, color=light, opacity=1.0,
                 pattern=dashed)
figure.draw_cone(figure.point_on_mirror -
                 2*incident_vector.dot(unit_normal)*unit_normal -
                 reflected_vector,
                 -reflected_vector, cone_size, color=light)


figure.side_view()
#mlab.view(azimuth=-7, elevation=90, distance=2)
#figure.mlab.view(-5.0, 95.0, 2.0, array([0.0, 0.29, 0.465]))
#figure.mlab.view (0.0, 90.0, 4.15, array([0.0, 0.29, 0.465]))
figure.mlab.view(0.0, 90.0, 4.15, array([-3.0, 0.29,  0.165]))
#import pdb; pdb.set_trace()
#figure.figure.scene.camera.zoom(2.0)
#camera_light = engine.scenes[0].scene.light_manager.lights[0]
#camera_light.activate = False
filename = os.path.splitext(sys.argv[0])[0]
figure.save(filename + ".tiff", magnification=1)
#save_figure(filename + ".eps", magnification=5)


