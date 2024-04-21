#include <Wire.h>
#include <LoRa.h>

#define   camAdd   0x21
#define   width     640
#define   height    480
#define   frameSize  width*height*2

uint8_t   rawImage[frameSize];

void  setup()
{
  Serial.begin(9600);
  while(!Serial);
  Wire.begin();
  writeToCamReg(0x12,0x80);
  delay(100);
  writeToCamReg(0x11,0x81);
  writeToCamReg(0x12,0x0C);
  while(!LoRa.begin(433E6));
  Serial.println("LoRa initialize");
  LoRa._onReceive(recieveData);
}

void  loop()
{
  
}

void  writeToCamReg(uint8_t add,uint8_t data)
{
 Wire.beginTransmission(camAdd);
 Wire.write(add);
 Wire.write(data);
 Wire.endTransmission();
}

void  recieveData()
{
  
}
