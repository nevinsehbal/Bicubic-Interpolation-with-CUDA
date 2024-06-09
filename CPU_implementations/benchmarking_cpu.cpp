#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../include/stb_image.h"
#include "../include/stb_image_write.h"
#include <stdint.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <fstream>

#define NUM_CHANNELS 3

double bicubic_weight(double t) {
    // Bicubic kernel function
    double A = -0.5;
    double abs_t = std::abs(t);
    double weight = 0;

    if (abs_t <= 1) {
        weight = (A + 2) * std::pow(abs_t, 3) - (A + 3) * std::pow(abs_t, 2) + 1;
    } else if (abs_t <= 2) {
        weight = A * std::pow(abs_t, 3) - 5 * A * std::pow(abs_t, 2) + 8 * A * abs_t - 4 * A;
    }

    return weight;
}

std::vector<std::vector<std::vector<int>>> bicubic_interpolation(std::vector<std::vector<std::vector<int>>> original_image, double scale_factor) {
    // Upscale dimensions
    int height = original_image.size();
    int width = original_image[0].size();
    int new_height = round(height * scale_factor);
    int new_width = round(width * scale_factor);

    // Initialize upscaled image
    std::vector<std::vector<std::vector<int>>> scaled_image(new_height, std::vector<std::vector<int>>(new_width, std::vector<int>(3, 0)));

    // Loop over each pixel in the upscaled image
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            // Convert coordinates from upscaled image to original image
            double original_x = x / scale_factor;
            double original_y = y / scale_factor;

            // Compute coordinates of the surrounding pixels
            int x1 = int(std::floor(original_x)) - 1;
            int y1 = int(std::floor(original_y)) - 1;

            // Compute fractional part of the coordinates
            double dx = original_x - x1 - 1;
            double dy = original_y - y1 - 1;

            // Initialize interpolated pixel values
            std::vector<double> interpolated_pixel(3, 0);

            // Apply bicubic interpolation kernel
            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < 4; ++i) {
                    // compute weights using bicubic kernel
                    double weight_x = bicubic_weight(dx - i);
                    double weight_y = bicubic_weight(dy - j);

                    // Compute pixel coordinates in the original image
                    int px = std::min(std::max(x1 + i, 0), width - 1);
                    int py = std::min(std::max(y1 + j, 0), height - 1);

                    // Compute interpolated pixel value
                    for (int c = 0; c < 3; ++c) {
                        interpolated_pixel[c] += weight_x * weight_y * original_image[py][px][c];
                    }
                }
            }
            // Assign interpolated pixel value to the upscaled image
            for (int c = 0; c < 3; ++c) {
                scaled_image[y][x][c] = std::min(std::max(int(interpolated_pixel[c]), 0), 255);
            }
        }
        if (y % 100 == 0) {
            std::cout << "Done: " << y << " rows out of " << new_height << std::endl;
        }
    }
    return scaled_image;
}

int main() {
    int width; // image width
    int height; // image height
    int bpp; // bytes per pixel if the image is RGB

    // Input and output paths
    std::vector<std::string> input_image_paths = {"../samples/butterfly.png", 
                                                    "../samples/cat.png",
                                                    "../samples/flower.png",
                                                    "../samples/bird.jpg",
                                                    "../samples/bridge.jpg",
                                                    "../samples/castle.jpg",
                                                    "../samples/forest.jpg",
                                                    "../samples/island.jpg",
                                                    "../samples/rosewater.jpg",
                                                    "../samples/lion.jpg"};
    std::vector<std::string> output_image_paths = {"../samples/cpu_cpp/butterfly_bicubic_interpolated.png",
                                                    "../samples/cpu_cpp/cat_bicubic_interpolated.png",
                                                    "../samples/cpu_cpp/flower_bicubic_interpolated.png",
                                                    "../samples/cpu_cpp/bird_bicubic_interpolated.png",
                                                    "../samples/cpu_cpp/bridge_bicubic_interpolated.png",
                                                    "../samples/cpu_cpp/castle_bicubic_interpolated.png",
                                                    "../samples/cpu_cpp/forest_bicubic_interpolated.png",
                                                    "../samples/cpu_cpp/island_bicubic_interpolated.png",
                                                    "../samples/cpu_cpp/rosewater_bicubic_interpolated.png",
                                                    "../samples/cpu_cpp/lion_bicubic_interpolated.png"};

    std::ofstream log_file("../samples/cpu_cpp/timings.txt");
    if (!log_file) {
        std::cerr << "Error opening log file!" << std::endl;
        return 1;
    }

    for (size_t i = 0; i < input_image_paths.size(); ++i) {
        const std::string& input_image_path = input_image_paths[i];
        const std::string& output_image_path = output_image_paths[i];

        auto start_read = std::chrono::high_resolution_clock::now();
        
        // Load image
        uint8_t* original_image = stbi_load(input_image_path.c_str(), &width, &height, &bpp, NUM_CHANNELS);
        if (!original_image) {
            std::cerr << "Failed to load image: " << input_image_path << std::endl;
            continue;
        }

        auto end_read = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> read_duration = end_read - start_read;

        std::cout << "Image loaded successfully. Width: " << width << " Height: " << height << " BPP: " << bpp << std::endl;

        // Convert the original_image pointer to a vector
        std::vector<std::vector<std::vector<int>>> original_image_vec(height, std::vector<std::vector<int>>(width, std::vector<int>(NUM_CHANNELS, 0)));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int c = 0; c < NUM_CHANNELS; ++c) {
                    original_image_vec[y][x][c] = original_image[(y * width + x) * NUM_CHANNELS + c];
                }
            }
        }

        std::cout << "Image converted to vector successfully." << std::endl;

        // Scale factor
        double scale_factor = 2.0;

        auto start_interpolation = std::chrono::high_resolution_clock::now();

        // Apply bicubic interpolation
        std::vector<std::vector<std::vector<int>>> interpolated_image = bicubic_interpolation(original_image_vec, scale_factor);

        auto end_interpolation = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> interpolation_duration = end_interpolation - start_interpolation;

        // Save the interpolated image
        int new_height = interpolated_image.size();
        int new_width = interpolated_image[0].size();
        uint8_t* interpolated_image_ptr = (uint8_t*)malloc(new_height * new_width * NUM_CHANNELS);
        for (int y = 0; y < new_height; ++y) {
            for (int x = 0; x < new_width; ++x) {
                for (int c = 0; c < NUM_CHANNELS; ++c) {
                    interpolated_image_ptr[(y * new_width + x) * NUM_CHANNELS + c] = interpolated_image[y][x][c];
                }
            }
        }

        auto start_write = std::chrono::high_resolution_clock::now();

        stbi_write_png(output_image_path.c_str(), new_width, new_height, NUM_CHANNELS, interpolated_image_ptr, new_width * NUM_CHANNELS);

        auto end_write = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> write_duration = end_write - start_write;

        // Free memory
        stbi_image_free(original_image);
        free(interpolated_image_ptr);

        // Log timings
        log_file << "Image: " << input_image_path << std::endl;
        log_file << "Read time: " << read_duration.count() << " seconds" << std::endl;
        log_file << "Interpolation time: " << interpolation_duration.count() << " seconds" << std::endl;
        log_file << "Write time: " << write_duration.count() << " seconds" << std::endl;
        log_file << std::endl;
    }

    log_file.close();
    return 0;
}
