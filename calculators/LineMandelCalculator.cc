/**
 * @file LineMandelCalculator.cc
 * @author David Chocholaty <xchoch09@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 2023-10-30
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>


#include "LineMandelCalculator.h"


LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
/*
    data  = (int *)(aligned_alloc(64, height * width * sizeof(int)));
    real_storage = (float *)(aligned_alloc(64, width * sizeof(float)));
    imag_storage = (float *)(aligned_alloc(64, width * sizeof(float)));
*/
    data  = (int *)(_mm_malloc(height * width * sizeof(int), 64));
    real_storage = (float *)(_mm_malloc(width * sizeof(float), 64));
    imag_storage = (float *)(_mm_malloc(width * sizeof(float), 64));
}

LineMandelCalculator::~LineMandelCalculator() {
/*
    free(data);
    data = NULL;

    free(imag_storage);
    imag_storage = NULL;

    free(real_storage);
    real_storage = NULL;
*/
    _mm_free(data);
    data = NULL;

    _mm_free(imag_storage);
    imag_storage = NULL;

    _mm_free(real_storage);
    real_storage = NULL;
}


int * LineMandelCalculator::calculateMandelbrot () {
    const int half_height = height / 2;

    // TODO otestovat jestli ponechat safelen
    #pragma omp simd simdlen(64) safelen(64)
    for (int i = 0; i <= half_height * width; i++) {
        data[i] = limit;
    }

    for (int i = 0; i <= half_height; i++) {
        const int row_start = i * width;

        const float y = y_start + i * dy; // current imaginary value

        #pragma omp simd simdlen(64) safelen(64)
        for (int j = 0; j < width; j++) {
            real_storage[j] = x_start + j * dx; // current real value
            imag_storage[j] = y;
        }

        int count = width;

        for (int k = 0; k < limit; k++) {

            #pragma omp simd reduction(-: count) simdlen(64) safelen(64)
            for (int j = 0; j < width; j++) {
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

        const int copy_row_start = (height - i - 1) * width;

        #pragma omp simd simdlen(64) safelen(64)
        for (int j = 0; j < width; j++) {
            data[copy_row_start + j] = data[row_start + j];
        }
    }

    return data;
}
