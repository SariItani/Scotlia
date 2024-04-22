#include <LoRa.h>

#define MAX_PACKET_SIZE 255 // Maximum size of LoRa packet
#define FRAME_SIZE (640 * 480) // Assuming fixed frame size

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize LoRa module
  if (!LoRa.begin(433E6)) {
    Serial.println("Error initializing LoRa");
    while (1);
  }
  Serial.println("LoRa initialized");

  // Start LoRa receiver
  LoRa.onReceive(receiveData);
  LoRa.receive();
}

void loop() {
  
}

void receiveData(int packetSize) {
  if (packetSize > 0) {
    // Receive data packet
    uint8_t receivedData[MAX_PACKET_SIZE];
    int bytesRead = LoRa.readBytes(receivedData, packetSize);

    // Process received data (e.g., decompress image)
    decompressAndTransmitToRaspberryPi(receivedData, bytesRead);
  }
}

/
void decompressAndTransmitToRaspberryPi(uint8_t* data, int size) {
  uint8_t decompressedData[FRAME_SIZE];
  decompressImage(data, decompressedData, size);
  transmitToRaspberryPi(decompressedData, FRAME_SIZE);
}

// Function to decompress image data (run-length decoding)
void decompressImage(uint8_t* inputData, uint8_t* outputData, int dataSize) {
  int outputIndex = 0;

  for (int i = 0; i < dataSize; i += 2) {
    uint8_t pixelValue = inputData[i];
    int count = inputData[i + 1];

    // Write the pixel value to the output data buffer 'count' times
    for (int j = 0; j < count; j++) {
      outputData[outputIndex++] = pixelValue;
    }
  }
}

void transmitToRaspberryPi(uint8_t* data, int size) {
  Serial.write(data, size);
}
