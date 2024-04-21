#include <Wire.h>
#include <LoRa.h>

#define OV7670_ADDRESS 0x21 // OV7670 I2C address
#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480
#define FRAME_SIZE (FRAME_WIDTH * FRAME_HEIGHT * 2) // Assuming RGB565 format

uint8_t imageData[FRAME_SIZE]; // Buffer for image data
uint8_t compressedData[FRAME_SIZE]; // Buffer for compressed image data

void writeToRegister(uint8_t regAddr, uint8_t data) {
  Wire.beginTransmission(OV7670_ADDRESS);
  Wire.write(regAddr);
  Wire.write(data);
  Wire.endTransmission();
}


int compressImage(uint8_t* inputData, uint8_t* outputData, int dataSize) {
  int outputIndex = 0;
  int count = 1;

  for (int i = 0; i < dataSize - 1; i++) {
    if (inputData[i] == inputData[i + 1]) {
      count++;
    } else {
      outputData[outputIndex++] = inputData[i];
      outputData[outputIndex++] = count;
      count = 1;
    }
  }

  // Handle the last pixel
  outputData[outputIndex++] = inputData[dataSize - 1];
  outputData[outputIndex++] = count;

  return outputIndex; // Return the size of the compressed data
}

void setup() {
  Serial.begin(9600);
  while (!Serial);
  delay(1000);
  Wire.begin();
  writeToRegister(0x12, 0x80); 
  delay(100); 
  writeToRegister(0x11, 0x81); 
  writeToRegister(0x12, 0x0C); 
  if (!LoRa.begin(433E6)) {
    Serial.println("Error initializing LoRa");
    while (1);
  }
  Serial.println("LoRa initialized");
  LoRa.onReceive(receiveData);
}

void loop() {
    capture();
    delay(5000);
}

void(receiveData)
{
  
}

void capture()
{
   writeToRegister(0x3A, 0x04); 
  writeToRegister(0x0A, 0x04);
  
  // Read image data from OV7670
  int index = 0;
  for (int i = 0; i < FRAME_HEIGHT; i++) {
    for (int j = 0; j < FRAME_WIDTH; j++) {
      // Read two bytes for RGB565 format
      Wire.requestFrom(OV7670_ADDRESS, 2);
      while (Wire.available() < 2);
      imageData[index++] = Wire.read(); // Red (5 bits)
      imageData[index++] = Wire.read(); // Green (6 bits) + Blue (5 bits)
    }
}


  int compressedSize = compressImage(imageData, compressedData, FRAME_SIZE);

  LoRa.beginPacket();
  LoRa.write(compressedData, compressedSize);
  LoRa.endPacket();

  Serial.println("Image transmitted");
}
