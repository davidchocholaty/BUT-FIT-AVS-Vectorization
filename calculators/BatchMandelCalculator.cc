/**
 * @file BatchMandelCalculator.cc
 * @author David Chocholaty <xchoch09@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date 2023-10-30
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>
#include <cmath>

#include "BatchMandelCalculator.h"


// Points of main cluter in mandel
/*
#define mainCluterIStart 0.33
#define mainCluterIEnd   0.67
#define mainCluterRStart 0.50
#define mainCluterREnd   0.73*/

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
    data  = (int *)(_mm_malloc(height * width * sizeof(int), 64));
    real_storage = (float *)(_mm_malloc(width * sizeof(float), 64));
    imag_storage = (float *)(_mm_malloc(width * sizeof(float), 64));
}

BatchMandelCalculator::~BatchMandelCalculator() {
    _mm_free(data);
    data = NULL;

    _mm_free(imag_storage);
    imag_storage = NULL;

    _mm_free(real_storage);
    real_storage = NULL;
}


int * BatchMandelCalculator::calculateMandelbrot () {
    constexpr int block_size = 64;
    constexpr float block_size_float = (float)block_size;
    const int half_height = height / 2;

    #pragma omp simd simdlen(64)
    for (int i = 0; i <= half_height * width; i++) {
        data[i] = limit;
    }

    //**********************
    /*
    const int mainGroupXBlockStart = std::ceil ((float) std::ceil (mainCluterRStart * width) / block_size) + 1;
    const int mainGroupXBlockEnd   = std::floor((float) std::floor(mainCluterREnd   * width) / block_size) - 1;
    const int mainGroupYStart      = std::ceil (mainCluterIStart * height);
    const int mainGroupYEnd        = std::floor(mainCluterIEnd   * height);*/
    //**********************

    // TODO pridat nebo aktualizovat pragmy z line
    // TODO zkusit aktualni verzi s vypoctem block_i_end a s predchozi verzi block_i_end
    // TODO zkusit porovnat s blockingem na height a bez blockingu na height
    for (int block_i = 0; block_i < std::ceil(half_height / block_size_float); block_i++) {
        const int block_i_start = block_i * block_size;
        const int block_i_end = ((block_i_start + block_size) >= half_height) ? half_height + 1 : block_i_start + block_size;

        for (int i = block_i_start; i < block_i_end; i++) {
        //  for (int i = 0; i <= half_height; i++) {
            const int row_start = i * width;

            const float y = y_start + i * dy; // current imaginary value

            #pragma omp simd simdlen(64)
            for (int j = 0; j < width; j++) {
                real_storage[j] = x_start + j * dx; // current real value
                imag_storage[j] = y;
            }

            for (int block_j = 0; block_j < std::ceil(width / block_size_float); block_j++) {
                //drop it is in bandelbrot set
                /*
                if (block_j >= mainGroupXBlockStart && block_j <= mainGroupXBlockEnd && i >= mainGroupYStart && i <= mainGroupYEnd) {
                    continue;
                }*/

                const int block_j_start = block_j * block_size;
                const int block_j_end = block_j_start + block_size;
                int count = block_size;

                for (int k = 0; k < limit; k++) {

                    #pragma omp simd reduction(-: count) simdlen(64)
                    for (int j = block_j_start; j < block_j_end; j++) {
                        if (data[row_start + j] == limit) {
                            const float r2 = real_storage[j] * real_storage[j];
                            const float i2 = imag_storage[j] * imag_storage[j];

                            if (r2 + i2 > 4.0f) {
                                data[row_start + j] = k;
                                --count;
                            } else {
                                imag_storage[j] = 2.0f * real_storage[j] * imag_storage[j] + y;
                                real_storage[j] = r2 - i2 + x_start + j * dx;
                            }
                        }
                    }

                    if (count == 0) {
                        break;
                    }
                }
            }

            const int copy_row_start = (height - i - 1) * width;

            #pragma omp simd simdlen(64)
            for (int j = 0; j < width; j++) {
                data[copy_row_start + j] = data[row_start + j];
            }
        }
    }

    return data;
}
