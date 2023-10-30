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
    data = (int *)(malloc(height * width * sizeof(int)));
    real_storage = (float *)(malloc(width * sizeof(float)));
    imag_storage = (float *)(malloc(width * sizeof(float)));
}

LineMandelCalculator::~LineMandelCalculator() {
    free(data);
    data = NULL;

    free(real_storage);
    real_storage = NULL;

    free(imag_storage);
    imag_storage = NULL;
}


int * LineMandelCalculator::calculateMandelbrot () {
    // TODO matice je symetricka, vypocet jen poloviny
    for (int i = 0; i < height * width; i++) {
        data[i] = 0;
    }

    for (int i = 0; i < height; i++) {
        float y = y_start + i * dy; // current imaginary value

        for (int k = 0; k < limit; k++) {
            for (int j = 0; j < width; j++) {
                float x = x_start + j * dx; // current real value

                float zReal = (k == 0) ? x : real_storage[j];
                float zImag = (k == 0) ? y : imag_storage[j];

                float r2 = zReal * zReal;
                float i2 = zImag * zImag;


                if (r2 + i2 <= 4.0f) {
                    real_storage[j] = r2 - i2 + x;
                    imag_storage[j] = 2.0f * zReal * zImag + y;
                    data[i * width + j] += 1;
                }
            }
        }
    }

	return data;
}
