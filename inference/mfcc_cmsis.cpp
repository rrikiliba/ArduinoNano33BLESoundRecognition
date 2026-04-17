#include "mfcc_cmsis.h"
#include <arm_math.h>
#include <math.h>

// generated headers
#include "mfcc_constants.h"
#include "model.h"

// CMSIS-DSP FFT instance
static arm_rfft_fast_instance_f32 fft_instance;

void mfcc_init() {
  // initialize hardware accelerated CMSIS-DSP fast FFT
  arm_rfft_fast_init_f32(&fft_instance, FRAME_SIZE);
}

void mfcc_compute(const int16_t* audio_buffer, TfLiteTensor* input_tensor) {
  float windowed_frame[FRAME_SIZE];
  float fft_output[FRAME_SIZE];
  float power_spectrum[FRAME_SIZE / 2 + 1];

  // 1. Calculate Global DC Offset
  float global_mean = 0.0f;
  float abs_sum = 0.0f;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    global_mean += (float)audio_buffer[i];
    abs_sum += fabs((float)audio_buffer[i]);
  }
  global_mean /= (float)BUFFER_SIZE;

  // uncomment this to check if the mic is working
  // if (abs_sum < 10.0f) {
  //   Serial.println("WARNING: Microphone signal is near zero!");
  // }

  // Auto Gain Control (AGC) - matching Python normalization
  float max_amp = 0.1f;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    float val = fabs((float)audio_buffer[i] - global_mean);
    if (val > max_amp) max_amp = val;
  }
  
  // prevent division by zero and match Python's dynamic range
  if (max_amp < 100.0f) max_amp = 100.0f; 
  float volume_scale = 1.0f / max_amp;

  int expected_frames = input_tensor->dims->data[1]; 

  for (int frame = 0; frame < expected_frames; frame++) {
    int start_idx = frame * HOP_SIZE;
    if (start_idx + FRAME_SIZE > BUFFER_SIZE) break;

    // Windowing
    for (int i = 0; i < FRAME_SIZE; i++) {
      float normalized_sample = ((float)audio_buffer[start_idx + i] - global_mean) * volume_scale;
      windowed_frame[i] = normalized_sample * hamming_window[i];
    }
    
    // FFT
    arm_rfft_fast_f32(&fft_instance, windowed_frame, fft_output, 0);
    
    // Power Spectrum
    power_spectrum[0] = fft_output[0] * fft_output[0];
    power_spectrum[FRAME_SIZE / 2] = fft_output[1] * fft_output[1];
    for (int i = 1; i < FRAME_SIZE / 2; i++) {
      float real = fft_output[2 * i];
      float imag = fft_output[2 * i + 1];
      power_spectrum[i] = (real * real) + (imag * imag);
    }
    
    // Mel Filterbank
    float mel_energies[NUM_MEL_BINS] = {0};
    for (int i = 0; i < NUM_MEL_BINS; i++) {
      for (int j = 0; j < (FRAME_SIZE / 2 + 1); j++) {
        mel_energies[i] += power_spectrum[j] * mel_matrix[j * NUM_MEL_BINS + i];
      }
    }
    
    // Log & DCT
    float mfcc[NUM_MFCC] = {0};
    for (int i = 0; i < NUM_MEL_BINS; i++) {
      // Add a small epsilon to prevent log(0) which causes -Infinity and breaks everything
      float log_mel = logf(mel_energies[i] + 1e-7f);
      for (int j = 0; j < NUM_MFCC; j++) {
        mfcc[j] += log_mel * dct_matrix[i * NUM_MFCC + j];
      }
    }
    
    for (int j = 0; j < NUM_MFCC; j++) {
      float scaled_val = (mfcc[j] - scaler_mean[j]) / (scaler_scale[j] + 1e-6f);
      
      // calculate quantization manually to ensure zero_point and scale are respected
      float q_val_float = (scaled_val / input_tensor->params.scale) + (float)input_tensor->params.zero_point;
      
      // hard clipping to INT8 range
      int32_t quantized_val = (int32_t)roundf(q_val_float);
      if (quantized_val < -128) quantized_val = -128;
      if (quantized_val > 127) quantized_val = 127;
      
      input_tensor->data.int8[frame * NUM_MFCC + j] = (int8_t)quantized_val;
    }
  }
}