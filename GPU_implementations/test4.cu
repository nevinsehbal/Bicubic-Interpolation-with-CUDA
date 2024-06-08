#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define NUM_CHANNELS 3

__constant__ double const_bicubic_weights[8];

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
    const int* __restrict__ original_image, int* __restrict__ scaled_image, 
    int original_width, int original_height, int new_width, int new_height, double scale_factor) {

    extern __shared__ int shared_image[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= new_width || y >= new_height) return;

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
                double weight_x = const_bicubic_weights[int(dx - i)];
                double weight_y = const_bicubic_weights[int(dy - j)];

                int px = min(max(x1 + i, 0), original_width - 1);
                int py = min(max(y1 + j, 0), original_height - 1);

                int pixel_value = original_image[(py * original_width + px) * NUM_CHANNELS + c];
                interpolated_pixel += weight_x * weight_y * pixel_value;
            }
        }
        scaled_image[(y * new_width + x) * NUM_CHANNELS + c] = min(max(int(interpolated_pixel), 0), 255);
    }
}

void precompute_weights() {
    double weights[8];
    for (int i = -3; i <= 4; ++i) {
        weights[i + 3] = bicubic_weight(i);
    }
    cudaMemcpyToSymbol(const_bicubic_weights, weights, 8 * sizeof(double));
}

int main(void) {
    int width, height, bpp;
    std::string input_image_path = "../samples/forest.jpg";
    uint8_t* original_image = stbi_load(input_image_path.c_str(), &width, &height, &bpp, NUM_CHANNELS);

    if (!original_image) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    std::cout << "Image loaded successfully. Width: " << width << " Height: " << height << " BPP: " << bpp << std::endl;

    int original_size = width * height * NUM_CHANNELS;
    int* h_original_image = (int*)malloc(original_size * sizeof(int));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < NUM_CHANNELS; ++c) {
                h_original_image[(y * width + x) * NUM_CHANNELS + c] = original_image[(y * width + x) * NUM_CHANNELS + c];
            }
        }
    }

    double scale_factor = 3.0;
    int new_width = round(width * scale_factor);
    int new_height = round(height * scale_factor);
    int new_size = new_width * new_height * NUM_CHANNELS;

    thrust::device_vector<int> d_original_image(h_original_image, h_original_image + original_size);
    thrust::device_vector<int> d_scaled_image(new_size, 0);

    precompute_weights();

    dim3 block_dim(16, 16);
    dim3 grid_dim((new_width + block_dim.x - 1) / block_dim.x, (new_height + block_dim.y - 1) / block_dim.y);

    size_t shared_mem_size = block_dim.x * block_dim.y * NUM_CHANNELS * sizeof(int);
    bicubic_interpolation_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        thrust::raw_pointer_cast(d_original_image.data()),
        thrust::raw_pointer_cast(d_scaled_image.data()),
        width, height, new_width, new_height, scale_factor
    );

    cudaDeviceSynchronize();

    thrust::host_vector<int> h_scaled_image = d_scaled_image;

    uint8_t* interpolated_image_ptr = (uint8_t*)malloc(new_size * sizeof(uint8_t));
    for (int i = 0; i < new_size; ++i) {
        interpolated_image_ptr[i] = static_cast<uint8_t>(h_scaled_image[i]);
    }

    std::string output_image_path = input_image_path.substr(0, input_image_path.find_last_of(".")) + "_bicubic_interpolated.png";
    stbi_write_png(output_image_path.c_str(), new_width, new_height, NUM_CHANNELS, interpolated_image_ptr, new_width * NUM_CHANNELS);

    stbi_image_free(original_image);
    free(h_original_image);
    free(interpolated_image_ptr);

    return 0;
}
/*
To optimize your bicubic interpolation code for GPU, you can consider several strategies that focus on enhancing memory access patterns, parallelization efficiency, and computational throughput. Here are some methods you can employ:

1. Shared Memory Usage
Utilize shared memory to cache the input pixel values that are repeatedly accessed by different threads within a block. This reduces global memory access latency significantly.

2. Optimize Memory Access Patterns
Ensure coalesced memory accesses to make the best use of the memory bandwidth. This can be achieved by aligning your memory accesses in a way that consecutive threads access consecutive memory locations.

3. Reduce Redundant Calculations
Pre-compute the weights and reuse them where possible. This can reduce the computational overhead.

4. Use Constant Memory for Weights
Store constant values such as bicubic weights in constant memory for faster access.

5. Use Texture Memory
Leverage CUDA's texture memory which is optimized for spatial locality, making it ideal for image processing tasks.

Key Optimizations Applied:
Pre-computed Weights in Constant Memory: The weights are precomputed and stored in constant memory to reduce redundant computations and speed up access.
Shared Memory: Shared memory is allocated for caching, though not fully utilized in this snippet. Further optimizations can fully exploit shared memory by caching image blocks.
Memory Access Patterns: Ensure memory accesses are coalesced for efficiency.
You can further experiment with these ideas and profile your application using tools like NVIDIA Nsight to identify bottlenecks and optimize accordingly.
*/