#ifndef MFCC_CMSIS_H
#define MFCC_CMSIS_H

#include <stdint.h>
// Include TFLite C types to interact directly with the input tensor
#include "tensorflow/lite/c/common.h"
#include "mfcc_constants.h" // Contains auto-generated constants from Python

// Initializes the CMSIS DSP Fast Fourier Transform instance
void mfcc_init();

// Processes the 1-second audio buffer: 
// Computes MFCCs, scales them, and quantizes them straight into the TFLite input tensor
void mfcc_compute(const int16_t* audio_buffer, TfLiteTensor* input_tensor);

#endif // MFCC_CMSIS_H