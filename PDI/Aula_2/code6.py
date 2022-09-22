# Import the module and the rotate and rescale functions
from PDI.src.pdi_utils import show_image,load_rotate_cat

image_cat = load_rotate_cat()
from skimage.transform import rescale, rotate

# Rotate the image 90 degrees clockwise
rotated_cat_image = rotate(image_cat, angle = -90.0)

# Rescale with anti aliasing
rescaled_with_aa = rescale(rotated_cat_image, 1/4, preserve_range = True, anti_aliasing = True)

# Rescale without anti aliasing
rescaled_without_aa = rescale(rotated_cat_image, 1/4, preserve_range = True, anti_aliasing = False)


# Show the resulting images
show_image(rescaled_with_aa, "Transformed with anti aliasing")
show_image(rescaled_without_aa, "Transformed without anti aliasing")
