#include <PDM.h>

// tensorFlow Lite Micro Includes
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// generated headers and external CMSIS implementation
#include "model.h"
#include "mfcc_cmsis.h"

// audio buffer for PDM data
// note that BUFFER_SIZE is defined in mfcc_constants.h (generated while training, included via mfcc_cmsis.h)
int16_t audio_buffer[BUFFER_SIZE];
volatile int buffer_index = 0;
volatile bool buffer_full = false;

// TFLite Micro globals
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// since the system is modular with regards to the addition of classes,
// remember to increase arena size if you add one and the inference starts going crazy
constexpr int kTensorArenaSize = 64 * 1024; 
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// PDM callback (called on new data)
// adds data to the global audio buffer until full
void onPDMdata() {
  int bytesAvailable = PDM.available();
  int16_t tempBuffer[256]; 
  
  if (bytesAvailable > sizeof(tempBuffer)) {
    bytesAvailable = sizeof(tempBuffer);
  }
  
  PDM.read(tempBuffer, bytesAvailable);
  int samples = bytesAvailable / 2;
  
  if (!buffer_full) {
    for(int i = 0; i < samples; i++) {
      audio_buffer[buffer_index++] = tempBuffer[i];
      if(buffer_index >= BUFFER_SIZE) {
        buffer_full = true;
        buffer_index = 0;
        break; 
      }
    }
  }
}

// initialization script
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000); 
  
  Serial.println("Initializing Audio Pipeline...");
  mfcc_init();

  Serial.println("Loading Model...");
  tflite_model = tflite::GetModel(model_tflite);
  if (tflite_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Error: Model schema mismatch!");
    while (1);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // check for silent failure before starting
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.print("Error: AllocateTensors() failed with status: ");
    Serial.println(allocate_status);
    Serial.println("Try increasing kTensorArenaSize!");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Verification: Ensure the model exported from Python matches our code's expectations
  Serial.print("Model Input Shape: ");
  Serial.print(input->dims->data[1]); // Frames
  Serial.print("x");
  Serial.println(input->dims->data[2]); // MFCCs


  Serial.println("Initializing Microphone...");
  PDM.onReceive(onPDMdata);
  PDM.setGain(20);
  
  if (!PDM.begin(1, SAMPLE_RATE)) {
    Serial.println("Error: Failed to start PDM!");
    while (1);
  }

  Serial.println("System Ready. Listening for audio...");
}


void loop() {
  if (buffer_full) {
    
    // extract features from the full audio buffer
    mfcc_compute(audio_buffer, input);

    // run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.print("Error: Inference Invoke failed with status: ");
      Serial.println(invoke_status);
      buffer_full = false;
      return;
    }

    int max_idx = 0;
    float max_prob = -1.0f;
    
    Serial.println("\n--- Predictions ---");
    for (int i = 0; i < NUM_CLASSES; i++) {
      // extract probability
      float prob = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
      
      Serial.print(GESTURE_LABELS[i]);
      Serial.print(": ");
      Serial.print(prob * 100.0f, 1);
      Serial.println("%");
      
      // determine winning class
      if (prob > max_prob) {
        max_prob = prob;
        max_idx = i;
      }
    }
    
    Serial.print(">> DETECTED: ");
    Serial.println(GESTURE_LABELS[max_idx]);
    Serial.println("-------------------");

    // reset buffer status to resume PDM data gathering
    buffer_full = false;
  }
}