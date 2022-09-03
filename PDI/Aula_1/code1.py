from PDI.src.pdi_utils import show_image

# Import the modules from skimage
from skimage import data, filters

# Load the rocket image
rocket = data.rocker()

# Convert the image to grayscale
gray_scaled_rocket = color.rgb2gray(rocker)

# Show the original image
show_image(rocket, 'Original RGB image')

# Show the grayscale image
show_image(gray_scaled_rocket, 'Grayscale image')