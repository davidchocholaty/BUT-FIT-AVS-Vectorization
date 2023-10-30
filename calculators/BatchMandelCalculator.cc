/**
 * @file BatchMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>

#include "BatchMandelCalculator.h"

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
    data = (int *)(malloc(height * width * sizeof(int)));
    real_storage = (float *)(malloc(width * sizeof(float)));
    imag_storage = (float *)(malloc(width * sizeof(float)));
}

BatchMandelCalculator::~BatchMandelCalculator() {
    free(data);
    data = NULL;

    free(real_storage);
    real_storage = NULL;

    free(imag_storage);
    imag_storage = NULL;
}


int * BatchMandelCalculator::calculateMandelbrot () {
	for (int i = 0; i < height * width; i++) {
        data[i] = 0;
    }

    constexpr int block_size = 64;

    for (int i = 0; i < height; i++) {
        float y = y_start + i * dy; // current imaginary value

        for (int k = 0; k < limit; k++) {
            for (int block_j = 0; block_j < width / block_size; block_j++) {
                for (int j = 0; j < block_size; j++) {
                    const int j_global = block_j * block_size + j;

                    float x = x_start + j_global * dx; // current real value

                    float zReal = (k == 0) ? x : real_storage[j_global];
                    float zImag = (k == 0) ? y : imag_storage[j_global];

                    float r2 = zReal * zReal;
                    float i2 = zImag * zImag;

                    if (r2 + i2 <= 4.0f) {
                        real_storage[j_global] = r2 - i2 + x;
                        imag_storage[j_global] = 2.0f * zReal * zImag + y;
                        data[i * width + j_global] += 1;
                    }
                }
            }
        }
    }

	return data;
}