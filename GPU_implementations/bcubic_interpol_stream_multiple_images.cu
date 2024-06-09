#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

#define NUM_CHANNELS 3

__device__ double bicubic_weight(double t) {
    double A = -0.5;
    double abs_t = fabs(t);
    double weight = 0;

    if (abs_t <= 1) {
        weight = (A + 2) * pow(abs_t, 3) - (A + 3) * pow(abs_t, 2) + 1;
    } else if (abs_t <= 2) {
        weight = A * pow(abs_t, 3) - 5 * A * pow(abs_t, 2) + 8 * A * abs_t - 4 * A;
    }

    return weight;
}

__global__ void bicubic_interpolation_kernel(
    const uint8_t* __restrict__ original_image, uint8_t* __restrict__ scaled_image, 
    int original_width, int original_height, int new_width, int new_height, double scale_factor) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < new_width && y < new_height) {
        double original_x = x / scale_factor;
        double original_y = y / scale_factor;

        int x1 = int(floor(original_x)) - 1;
        int y1 = int(floor(original_y)) - 1;

        double dx = original_x - x1 - 1;
        double dy = original_y - y1 - 1;

        for (int c = 0; c < NUM_CHANNELS; ++c) {
            double interpolated_pixel = 0;

            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < 4; ++i) {
                    double weight_x = bicubic_weight(dx - i);
                    double weight_y = bicubic_weight(dy - j);

                    int px = min(max(x1 + i, 0), original_width - 1);
                    int py = min(max(y1 + j, 0), original_height - 1);

                    int pixel_value = original_image[(py * original_width + px) * NUM_CHANNELS + c];
                    interpolated_pixel += weight_x * weight_y * pixel_value;
                }
            }
            scaled_image[(y * new_width + x) * NUM_CHANNELS + c] = min(max(int(interpolated_pixel), 0), 255);
        }
    }
}

int main(void) {
    /*
    * Load images
    */
    std::vector<std::string> input_image_paths = {
        "../samples/butterfly.png",
        "../samples/cat.png",
        "../samples/flower.png",
        "../samples/bird.jpg",
        "../samples/bridge.jpg",
        "../samples/castle.jpg",
        "../samples/forest.jpg",
        "../samples/island.jpg",
        "../samples/lion.jpg",
        "../samples/rosewater.jpg"};

    std::vector<std::string> output_image_paths = {
        "../samples/gpu_raw_streams/butterfly.png",
        "../samples/gpu_raw_streams/cat.png",
        "../samples/gpu_raw_streams/flower.png",
        "../samples/gpu_raw_streams/bird.jpg",
        "../samples/gpu_raw_streams/bridge.jpg",
        "../samples/gpu_raw_streams/castle.jpg",
        "../samples/gpu_raw_streams/forest.jpg",
        "../samples/gpu_raw_streams/island.jpg",
        "../samples/gpu_raw_streams/lion.jpg",
        "../samples/gpu_raw_streams/rosewater.jpg"};

    int num_images = input_image_paths.size();

    std::vector<int> widths(num_images);
    std::vector<int> heights(num_images);
    std::vector<int> bpps(num_images);
    std::vector<uint8_t*> original_images(num_images);

    for (int i = 0; i < num_images; ++i) {
        original_images[i] = stbi_load(input_image_paths[i].c_str(), &widths[i], &heights[i], &bpps[i], NUM_CHANNELS);
        if (!original_images[i]) {
            std::cerr << "Error loading image: " << input_image_paths[i] << std::endl;
            return -1;
        }
        std::cout << "Image loaded successfully: " << input_image_paths[i] << ". Width: " << widths[i] << " Height: " << heights[i] << " BPP: " << bpps[i] << std::endl;
    }

    double scale_factor = 2.0;
    std::vector<int> new_widths(num_images);
    std::vector<int> new_heights(num_images);
    std::vector<int> original_sizes(num_images);
    std::vector<int> new_sizes(num_images);

    for (int i = 0; i < num_images; ++i) {
        new_widths[i] = round(widths[i] * scale_factor);
        new_heights[i] = round(heights[i] * scale_factor);
        original_sizes[i] = widths[i] * heights[i] * NUM_CHANNELS;
        new_sizes[i] = new_widths[i] * new_heights[i] * NUM_CHANNELS;
    }

    std::vector<cudaStream_t> streams(num_images);
    for (int i = 0; i < num_images; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    std::vector<thrust::device_vector<uint8_t>> d_original_images(num_images);
    std::vector<thrust::device_vector<uint8_t>> d_scaled_images(num_images);
    for (int i = 0; i < num_images; ++i) {
        d_original_images[i] = thrust::device_vector<uint8_t>(original_images[i], original_images[i] + original_sizes[i]);
        d_scaled_images[i] = thrust::device_vector<uint8_t>(new_sizes[i], 0);
    }

    dim3 block_dim(16, 16);
    std::vector<dim3> grid_dims(num_images);
    for (int i = 0; i < num_images; ++i) {
        grid_dims[i] = dim3((new_widths[i] + block_dim.x - 1) / block_dim.x, (new_heights[i] + block_dim.y - 1) / block_dim.y);
    }

    std::vector<cudaEvent_t> start_events(num_images);
    std::vector<cudaEvent_t> stop_events(num_images);
    for (int i = 0; i < num_images; ++i) {
        cudaEventCreate(&start_events[i]);
        cudaEventCreate(&stop_events[i]);
    }

    for (int i = 0; i < num_images; ++i) {
        cudaEventRecord(start_events[i], streams[i]);
        bicubic_interpolation_kernel<<<grid_dims[i], block_dim, 0, streams[i]>>>(
            thrust::raw_pointer_cast(d_original_images[i].data()),
            thrust::raw_pointer_cast(d_scaled_images[i].data()),
            widths[i], heights[i], new_widths[i], new_heights[i], scale_factor
        );
        cudaEventRecord(stop_events[i], streams[i]);
    }

    for (int i = 0; i < num_images; ++i) {
        cudaEventSynchronize(stop_events[i]);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_events[i], stop_events[i]);
        std::cout << "Time taken to interpolate image " << i << ": " << milliseconds << " ms" << std::endl;
    }

    std::vector<thrust::host_vector<uint8_t>> h_scaled_images(num_images);
    for (int i = 0; i < num_images; ++i) {
        cudaEventRecord(start_events[i], streams[i]);
        h_scaled_images[i] = d_scaled_images[i];
        cudaEventRecord(stop_events[i], streams[i]);
    }

    for (int i = 0; i < num_images; ++i) {
        cudaEventSynchronize(stop_events[i]);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_events[i], stop_events[i]);
        std::cout << "Time taken to copy data back to host for image " << i << ": " << milliseconds << " ms" << std::endl;
    }

    for (int i = 0; i < num_images; ++i) {
        const uint8_t* image_data_ptr = h_scaled_images[i].data();
        auto cpu_start = std::chrono::high_resolution_clock::now();

        stbi_write_png(output_image_paths[i].c_str(), new_widths[i], new_heights[i], NUM_CHANNELS, image_data_ptr, new_widths[i] * NUM_CHANNELS);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
        std::cout << "Time taken to write image " << i << " to disk: " << cpu_duration.count() << " ms" << std::endl;
    }

    for (int i = 0; i < num_images; ++i) {
        stbi_image_free(original_images[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(start_events[i]);
        cudaEventDestroy(stop_events[i]);
    }

    return 0;
}
