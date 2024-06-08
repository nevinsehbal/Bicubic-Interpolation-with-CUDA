# Bicubic-Interpolation-with-CUDA

Bicubic interpolation algorithm implementation on CUDA to upscale images, also referred to as superresolution

The directory contents are described below.

* CPU_implementations
  * bcubic_interpol_cpu.cpp --> CPU implementation of bicubic interpolation algorithm with c++. Large images takes long time during processing.
  * bcubic_interp_cpu.py --> CPU implementation of bicubic interpolation algorithm with Python. Large images takes long time during processing.
* include
  * stb_image_write.h, stb_image.h --> image file read/write operations management
* samples --> example images with varying resolutions

To build and run bcubic_interpol_cpu.cpp:

```
g++ -std=c++11 bcubic_interpol_cpu.cpp -o executable
```

To run bcubic_interpol_cpu.py:

* Numpy Version:  1.26.4
* OpenCV Version:  4.9.0

# Reference

[Bicubic-interpolation implementation by https://github.com/rootpine/](https://github.com/rootpine/Bicubic-interpolation)
