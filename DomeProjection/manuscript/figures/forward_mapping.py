from numpy import array
from mayavi import mlab
import sys, os

from dome_figure import DomeFigure


# Create the figure and pick colors for each object.
figure = DomeFigure(size=(1560,830))
mirror = figure.colors['silver']
projector = figure.colors['light_gray']
ball = figure.colors['white']
dome = figure.colors['silver']
light = figure.colors['red']
triangle1 = figure.colors['green']
triangle2 = figure.colors['blue']
triangle3 = figure.colors['black']
mouse = figure.colors['black']
# Specify line properties.
dashed = (0.01, 0.01)
line_width = 3

""" Draw the mirror, dome, projector, track ball and mouse. """
# draw the mirror
figure.draw_mirror(color=mirror, opacity=0.25)
# draw the dome
figure.draw_dome(color=dome, opacity=0.25)
# draw the track ball
figure.draw_ball(color=ball, opacity=0.5)
# draw the mouse
figure.draw_mouse(color=mouse, opacity=0.3)
# draw a 3D extrusion for the projector
figure.draw_projector(color=projector, opacity=0.25)

""" Draw the light from a single point in the projector's image reflecting off
the mirror and hitting the dome. """
# draw a line from the projector's focal point to the point on the mirror
figure.draw_line(figure.focal_point, figure.point_on_mirror,
                 line_width=line_width, color=light, opacity=1.0)
# draw a line from the point on the mirror to the point on the dome
figure.draw_line(figure.point_on_mirror, figure.point_on_dome,
                 line_width=line_width, color=light, opacity=1.0)

""" Draw the lines for the triangle consisting of the projector's focal point,
the mirror's center, and a point on the mirror. """
# draw line from center of mirror to projector's focal point
figure.draw_line(figure.mirror_center, figure.focal_point,
                 line_width=line_width, color=triangle1, opacity=1.0,
                 pattern=dashed)
# draw a line from the mirror's center to a point on the mirror
figure.draw_line(figure.mirror_center, figure.point_on_mirror, 
                 line_width=line_width, color=triangle1, opacity=1.0,
                 pattern=dashed)

""" Draw the lines for the triangle consisting of a point on the mirror, the
dome center, and a point on the dome. """
# draw a line from the dome center to the point on the dome
figure.draw_line(figure.dome_center, figure.point_on_dome,
                 line_width=line_width, color=triangle2, opacity=1.0,
                 pattern=dashed)
# draw a line from the animal's position to the point on the mirror
figure.draw_line(figure.dome_center, figure.point_on_mirror,
                 line_width=line_width, color=triangle2, opacity=1.0,
                 pattern=dashed)

""" Draw the lines for the triangle consisting of the mirror center, a point on
the dome, and the animal's position. """
# draw a line from the mirror center to the point on the dome
figure.draw_line(figure.mirror_center, figure.point_on_dome,
                 line_width=line_width, color=triangle3, opacity=1.0,
                 pattern=dashed)
# draw a line from the mirror center to the animal's position
figure.draw_line(figure.mirror_center, figure.animal_position,
                 line_width=line_width, color=triangle3, opacity=1.0,
                 pattern=dashed)
# draw a line from the animal's position to the point on the dome
figure.draw_line(figure.animal_position, figure.point_on_dome,
                 line_width=line_width, color=triangle3, opacity=1.0,
                 pattern=dashed)

figure.side_view()
#mlab.view(azimuth=-7, elevation=90, distance=2)
#figure.mlab.view(-5.0, 95.0, 2.0, array([0.0, 0.29, 0.465]))
figure.mlab.view (0.0, 90.0, 4.15, array([0.0, 0.29, 0.465]))
figure.figure.scene.camera.zoom(2.0)
#camera_light = engine.scenes[0].scene.light_manager.lights[0]
#camera_light.activate = False
#import pdb; pdb.set_trace()
filename = os.path.splitext(sys.argv[0])[0]
figure.save(filename + ".tiff", magnification=1)
#save_figure(filename + ".eps", magnification=5)


