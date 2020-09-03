#include "DFRobot_BMP388.h"
#include "DFRobot_BMP388_I2C.h"
#include "Wire.h"
#include "SPI.h"
#include "math.h"
#include "bmp3_defs.h"


#include <DFRobot_BMX160.h>


/*If there is no need to calibrate altitude, comment this line*/
#define CALIBRATE_Altitude

/*Create a bmp388 object to communicate with IIC.*/
DFRobot_BMP388_I2C bmp388;

float seaLevel;

DFRobot_BMX160 bmx160;


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  delay(100);
  
  //init the hardware bmx160  
  if (bmx160.begin() != true){
    Serial.println("init false");
    while(1);
  }

  bmp388.set_iic_addr(0x76);
  /* Initialize bmp388*/
  while(bmp388.begin()){
    Serial.println("Initialize error!");
    delay(1000);
  }

  seaLevel = bmp388.readSeaLevel(161.5);
  /*La Mesa is 161.5m */
  Serial.print("seaLevel : ");
  Serial.print(seaLevel);
  Serial.println(" Pa");
  
}

void loop() {
  // put your main code here, to run repeatedly:


  /*#ifdef CALIBRATE_Altitude*/
  /* Read the calibrated altitude */
  /*float altitude = bmp388.readCalibratedAltitude(seaLevel);*/
  /*Serial.print("calibrate Altitude : ");
  Serial.print(altitude);
  Serial.println(" m");
  #else
  */
  /* Read the altitude */
  /*float altitude = bmp388.readAltitude();
  Serial.print("Altitude : ");
  Serial.print(altitude);
  Serial.println(" m");
  #endif
  delay(100);*/



  bmx160SensorData Omagn, Ogyro, Oaccel;

  /* Get a new sensor event */
  bmx160.getAllData(&Omagn, &Ogyro, &Oaccel);

  /* Display the accelerometer results (accelerometer data is in m/s^2) */
  Serial.print("A ");
  Serial.print("X: "); Serial.print(Oaccel.x    ); Serial.print("  ");
  Serial.print("Y: "); Serial.print(Oaccel.y    ); Serial.print("  ");
  Serial.print("Z: "); Serial.print(Oaccel.z    ); Serial.print("  ");
  Serial.println("m/s^2");

  

  delay(10);
}
