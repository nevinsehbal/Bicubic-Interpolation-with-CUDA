import cv2
import time

def resize_images_with_timing(input_file_paths, output_file_paths, scale_ratio):
    if len(input_file_paths) != len(output_file_paths):
        raise ValueError("Number of input and output files must be the same.")
    
    for input_file_path, output_file_path in zip(input_file_paths, output_file_paths):
        # Measure the time to read the image
        start_time = time.time()
        image = cv2.imread(input_file_path)
        if image is None:
            raise FileNotFoundError(f"No image found at {input_file_path}")
        read_time = time.time() - start_time

        # Get the dimensions of the image
        height, width = image.shape[:2]

        # Calculate the new dimensions
        new_width = int(width * scale_ratio)
        new_height = int(height * scale_ratio)

        # Measure the time to resize the image
        start_time = time.time()
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        resize_time = time.time() - start_time

        # Measure the time to write the resized image
        start_time = time.time()
        cv2.imwrite(output_file_path, resized_image)
        write_time = time.time() - start_time

        # Print the timings
        print(f"Original image dimensions: {width}x{height}")
        print(f"Filename: {input_file_path}")
        print(f"Time to read the image: {read_time:.6f} seconds")
        print(f"Time to resize the image: {resize_time:.6f} seconds")
        print(f"Time to write the image: {write_time:.6f} seconds")
        print("")


# Example usage
input_image_paths = [
    '../samples/butterfly.png',
    '../samples/cat.png',
    '../samples/flower.png',
    '../samples/bird.jpg',
    '../samples/bridge.jpg',
    '../samples/castle.jpg',
    '../samples/forest.jpg',
    '../samples/island.jpg',
    '../samples/lion.jpg',
    '../samples/rosewater.jpg'
]

output_image_paths = [
    '../samples/cpu_cv2_python/butterfly_upscaled.png',
    '../samples/cpu_cv2_python/cat_upscaled.png',
    '../samples/cpu_cv2_python/flower_upscaled.png',
    '../samples/cpu_cv2_python/bird_upscaled.png',
    '../samples/cpu_cv2_python/bridge_upscaled.png',
    '../samples/cpu_cv2_python/castle_upscaled.png',
    '../samples/cpu_cv2_python/forest_upscaled.png',
    '../samples/cpu_cv2_python/island_upscaled.png',
    '../samples/cpu_cv2_python/lion_upscaled.png',
    '../samples/cpu_cv2_python/rosewater_upscaled.png'
]
scale_ratio = 2.0  # For example, resizing to double the original size

resize_images_with_timing(input_image_paths, output_image_paths, scale_ratio)
