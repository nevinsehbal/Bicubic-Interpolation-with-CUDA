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


int main() {
int width; //image width
    int height; //image height
    int bpp;  //bytes per pixel if the image is RGB

    std::string input_image_path = "../samples/butterfly_bicubic_interpolated_cpu.png";
    
    // Load image
    uint8_t* original_image = stbi_load(input_image_path.c_str(), &width, &height, &bpp, NUM_CHANNELS);

    std::cout<<"Image loaded successfully. Width: "<<width<<" Height: "<<height<<" BPP: "<<bpp<<std::endl;

    uint8_t *original_image1 = stbi_load("../samples/butterfly_bicubic_interpolated_gpu.png", &width, &height, &bpp, NUM_CHANNELS);

    std::cout<<"Image loaded successfully. Width: "<<width<<" Height: "<<height<<" BPP: "<<bpp<<std::endl;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < NUM_CHANNELS; ++c) {
                if(original_image1[(y * width + x) * NUM_CHANNELS + c] != original_image[(y * width + x) * NUM_CHANNELS + c]){
                    std::cout<<"Images are different at pixel: "<<x<<", "<<y<<", "<<c<<std::endl;
                    std::cout<<"Original image: "<<(int)original_image[(y * width + x) * NUM_CHANNELS + c]<<std::endl;
                    std::cout<<"Original1 image: "<<(int)original_image1[(y * width + x) * NUM_CHANNELS + c]<<std::endl;
                    return 1;
                }
            }
        }
    }
    std::cout<<"Images are the same"<<std::endl;
    // Free memory
    stbi_image_free(original_image);
    stbi_image_free(original_image1);
    // stbi_image_free(original_image2);

    return 0;
}