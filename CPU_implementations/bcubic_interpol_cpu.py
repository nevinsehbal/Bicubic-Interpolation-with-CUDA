import numpy as np
import cv2

print("Numpy Version: ", np.__version__)
print("OpenCV Version: ", cv2.__version__)  

def bicubic_weight(t):
    # Bicubic kernel function
    A = -0.5
    abs_t = np.abs(t)
    if abs_t <= 1:
        weight = (A + 2) * abs_t**3 - (A + 3) * abs_t**2 + 1
    elif abs_t <= 2:
        weight = A * abs_t**3 - 5 * A * abs_t**2 + 8 * A * abs_t - 4 * A
    else:
        weight = 0
    return weight

def bicubic_interpolation(original_image, scale_factor):
    # Upscale dimensions
    height, width, _ = original_image.shape
    new_height = round(height * scale_factor)
    new_width = round(width * scale_factor)

    # Initialize upscaled image
    scaled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Loop over each pixel in the upscaled image
    for y in range(new_height):
        for x in range(new_width):
            # Convert coordinates from upscaled image to original image
            original_x = x / scale_factor
            original_y = y / scale_factor

            # Compute coordinates of surrounding pixels
            x1 = int(np.floor(original_x)) - 1
            y1 = int(np.floor(original_y)) - 1

            # Compute fractional parts
            dx = original_x - x1 - 1
            dy = original_y - y1 - 1

            # Initialize interpolated pixel values
            interpolated_pixel = np.zeros(3, dtype=np.float64)

            # Apply bicubic interpolation kernel
            for j in range(4):
                for i in range(4):
                    # Compute weights using bicubic kernel
                    weight_x = bicubic_weight(dx - i)
                    weight_y = bicubic_weight(dy - j)

                    # Compute pixel coordinates in original image
                    px = min(max(x1 + i, 0), width - 1)
                    py = min(max(y1 + j, 0), height - 1)


                    # Compute interpolated pixel value
                    interpolated_pixel += weight_x * weight_y * original_image[py, px]

            # Assign interpolated pixel value to upscaled image
            scaled_image[y, x] = np.clip(interpolated_pixel, 0, 255).astype(np.uint8)
        print("Progress: {:.2f}%".format((y + 1) / new_height * 100), end='\r')

    return scaled_image

input_image_path = '../samples/forest.jpg'

# Load the original image
original_image = cv2.imread(input_image_path)

if(original_image is None):
    print("Image not found")
    exit()

# Scale factor for upscaling
scale_factor = 2

# Perform bicubic interpolation
scaled_image = bicubic_interpolation(original_image, scale_factor)

print(scaled_image.shape, original_image.shape)

# Save the upscaled image
output_image_path = '../samples/py_raw_cpu_forest_upscaled.png'
cv2.imwrite(output_image_path, scaled_image)