#include "DFRobot_BMP388.h"
#include "DFRobot_BMP388_I2C.h"
#include "Wire.h"
#include "SPI.h"
#include "math.h"
#include "bmp3_defs.h"
#include <DFRobot_BMX160.h>

float thetaM;
float phiM;
float theta;
float phi;
float dt;
float Xm;
float Ym;
float psi;
unsigned long millisOld;


/*If there is no need to calibrate altitude, comment this line*/
#define CALIBRATE_Altitude

/*Create a bmp388 object to communicate with IIC.*/
DFRobot_BMP388_I2C bmp388;

float seaLevel;

DFRobot_BMX160 bmx160;


void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  delay(1000);
  millisOld=millis();
  
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
}

void loop() {

  bmx160SensorData Omagn, Ogyro, Oaccel;

  /* Get a new sensor event */
  bmx160.getAllData(&Omagn, &Ogyro, &Oaccel);

  /* Calculate Angles */
  thetaM = -atan2(Oaccel.x,Oaccel.z);
  phiM = -atan2(Oaccel.y, Oaccel.z);

  dt=(millis()-millisOld)/1000.;
  millisOld=millis();

  theta = (theta +Ogyro.y*dt)*.5 + thetaM*.5;
  phi=(phi-Ogyro.x*dt)*.5 + phiM*.5;

  
  Xm = Omagn.x*cos(theta)-Omagn.y*sin(phi)*sin(theta)+Omagn.z*cos(phi)*sin(theta);
  Ym = Omagn.y*cos(phi)+Omagn.z*sin(phi);
  psi = atan2(Ym, Xm);

  
  /* Display the accelerometer results (accelerometer data is in m/s^2) */
  Serial.print(Oaccel.x); Serial.print(",");
  Serial.print(Oaccel.y); Serial.print(",");
  Serial.print(Oaccel.z); Serial.print(",");

  /* Display the gyroscope results (gyroscope data is in g) */
  Serial.print(Ogyro.x); Serial.print(",");
  Serial.print(Ogyro.y); Serial.print(",");
  Serial.print(Ogyro.z); Serial.print(",");
  
  /* Display the magnetometer results (magn is magnetometer in uTesla) */
  Serial.print(Omagn.x); Serial.print(",");
  Serial.print(Omagn.y); Serial.print(",");
  Serial.print(Omagn.z); Serial.print(",");

  /* Display Euler Angles */
  Serial.print(phi); Serial.print(",");
  Serial.print(theta); Serial.print(",");
  Serial.print(psi); Serial.print(",");
  
  /* Read the altitude */
  float altitude = bmp388.readAltitude();
  Serial.println(altitude);
  Serial.println();
  delay(50);
}
