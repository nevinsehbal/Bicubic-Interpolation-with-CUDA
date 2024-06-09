import numpy as np
import cv2
import time

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

def process_images(input_image_paths, output_image_paths, scale_factor, log_file):
    with open(log_file, 'w') as log:
        for input_image_path, output_image_path in zip(input_image_paths, output_image_paths):
            log.write(f"Processing {input_image_path}...\n")
            log.flush()
            
            # Measure file read time
            read_start_time = time.time()
            # Load the original image
            original_image = cv2.imread(input_image_path)
            read_time = time.time() - read_start_time
            
            if original_image is None:
                log.write(f"Error: Image {input_image_path} not found\n")
                log.flush()
                continue

            # Perform bicubic interpolation and measure time
            start_time = time.time()
            scaled_image = bicubic_interpolation(original_image, scale_factor)
            interpolation_time = time.time() - start_time

            # Measure file write time
            write_start_time = time.time()
            # Save the upscaled image
            cv2.imwrite(output_image_path, scaled_image)
            write_time = time.time() - write_start_time

            log.write(f"Input image: {input_image_path}\n")
            log.write(f"Output image: {output_image_path}\n")
            log.write(f"File read time: {read_time:.2f} seconds\n")
            log.write(f"Interpolation time: {interpolation_time:.2f} seconds\n")
            log.write(f"File write time: {write_time:.2f} seconds\n\n")
            log.flush()

# Example usage
input_image_paths = [
    # '../samples/butterfly.png',
    # '../samples/cat.png',
    # '../samples/flower.png',
    # '../samples/bird.jpg',
    # '../samples/bridge.jpg',
    # '../samples/castle.jpg',
    # '../samples/forest.jpg',
    # '../samples/island.jpg',
    # '../samples/lion.jpg',
    # '../samples/rosewater.jpg'
]

output_image_paths = [
    # '../samples/cpu_raw_python/butterfly_upscaled.png',
    # '../samples/cpu_raw_python/cat_upscaled.png',
    # '../samples/cpu_raw_python/flower_upscaled.png',
    # '../samples/cpu_raw_python/bird_upscaled.png',
    # '../samples/cpu_raw_python/bridge_upscaled.png',
    # '../samples/cpu_raw_python/castle_upscaled.png',
    # '../samples/cpu_raw_python/forest_upscaled.png',
    # '../samples/cpu_raw_python/island_upscaled.png',
    # '../samples/cpu_raw_python/lion_upscaled.png',
    # '../samples/cpu_raw_python/rosewater_upscaled.png'
]

scale_factor = 2
log_file = '../samples/cpu_raw_python/interpolation_log.txt'

process_images(input_image_paths, output_image_paths, scale_factor, log_file)
