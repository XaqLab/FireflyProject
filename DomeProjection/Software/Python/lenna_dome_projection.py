#!/usr/local/bin/python
# written for python 2.7.8

############################################################
# Main Program Starts Here
############################################################
from PIL import Image
from dome_projection import DomeProjection

if __name__ == "__main__":
    input_image = Image.open("Lenna.png")
    
    # read pixel data from image into a list
    if input_image.mode == 'RGB':
        print "RGB image"
        print input_image.size
    elif input_image.mode == 'L':
        print "Grayscale image"
        print input_image.size
    else:
        print "Unsupported image mode:", input_image.mode
        exit()
    
    # show original image
    input_image.show()
    
    print "Creating instance of DomeProjection Class"
    dome = DomeProjection()
    print "Done initializing dome"
    
    output_image = dome.warp_image_for_dome(input_image)
    
    output_image.show()
    #output_image.save("lenna_warped.png", "PNG")


