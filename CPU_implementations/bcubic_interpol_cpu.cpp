#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../include/stb_image.h"
#include "../include/stb_image_write.h"
#include <stdint.h>
#include <algorithm>

#define NUM_CHANNELS 3
#include <iostream>

#include <cmath>
#include <vector>
#include <iostream>

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

void bicubic_interpolation2(uint8_t* original_image, uint8_t* scaled_image, int width, int height, int new_width, int new_height, double scale_factor) {
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            double original_x = x / scale_factor;
            double original_y = y / scale_factor;
            int x1 = int(std::floor(original_x)) - 1;
            int y1 = int(std::floor(original_y)) - 1;
            double dx = original_x - x1 - 1;
            double dy = original_y - y1 - 1;
            double interpolated_pixel[NUM_CHANNELS] = {0};

            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < 4; ++i) {
                    double weight_x = bicubic_weight(dx - i);
                    double weight_y = bicubic_weight(dy - j);
                    int px = std::min(std::max(x1 + i, 0), width - 1);
                    int py = std::min(std::max(y1 + j, 0), height - 1);

                    for (int c = 0; c < NUM_CHANNELS; ++c) {
                        interpolated_pixel[c] += weight_x * weight_y * original_image[(py * width + px) * NUM_CHANNELS + c];
                    }
                }
            }

            for (int c = 0; c < NUM_CHANNELS; ++c) {
                scaled_image[(y * new_width + x) * NUM_CHANNELS + c] = std::min(std::max(int(interpolated_pixel[c]), 0), 255);
            }
        }
        if (y % 100 == 0) {
            std::cout << "Done: " << y << " rows out of " << new_height << std::endl;
        }
    }
}

int main() {
    int width, height, bpp;
    std::string input_image_path = "C:/workspace/sehbal-ws/Bicubic-Interpolation-with-CUDA/samples/island.jpg";
    uint8_t* original_image = stbi_load(input_image_path.c_str(), &width, &height, &bpp, NUM_CHANNELS);

    if (!original_image) {
        std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
        return -1;
    }

    std::cout << "Image loaded successfully. Width: " << width << " Height: " << height << " BPP: " << bpp << std::endl;

    double scale_factor = 1.0;
    int new_width = round(width * scale_factor);
    int new_height = round(height * scale_factor);

    try {
        uint8_t* scaled_image = new uint8_t[new_height * new_width * NUM_CHANNELS];

        bicubic_interpolation2(original_image, scaled_image, width, height, new_width, new_height, scale_factor);

        std::cout << "Interpolation done successfully." << std::endl;

        std::string output_image_path = input_image_path.substr(0, input_image_path.find_last_of(".")) + "_bicubic_interpolated_cpp_test.png";
        stbi_write_png(output_image_path.c_str(), new_width, new_height, NUM_CHANNELS, scaled_image, new_width * NUM_CHANNELS);

        std::cout << "Image saved successfully." << std::endl;

        delete[] scaled_image;
        stbi_image_free(original_image);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        stbi_image_free(original_image);
        return -1;
    }

    return 0;
}