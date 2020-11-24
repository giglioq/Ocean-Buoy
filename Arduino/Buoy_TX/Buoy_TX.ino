/*

Demonstrates sending and receiving packets of different length.

Radio    Arduino
CE    -> 7
CSN   -> 8 (Hardware SPI SS)
MOSI  -> 11 (Hardware SPI MOSI)
MISO  -> 12 (Hardware SPI MISO)
SCK   -> 13 (Hardware SPI SCK)
IRQ   -> No connection
VCC   -> No more than 3.6 volts
GND   -> GND

*/

#include <SPI.h>
#include <NRFLite.h>

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

/* Set the delay between fresh samples */
uint16_t BNO055_SAMPLERATE_DELAY_MS = 100;

#define TCAADDR 0x70

/* select I2C channel using TCA9548A multiplexer */
void tcaselect(uint8_t channel)
{
  Serial.print("I2C Channel: ");  Serial.println(channel);
  Wire.beginTransmission(0x70);
  Wire.write(1 << channel);
  Wire.endTransmission();
}


// Check I2C device address and correct line below (by default address is 0x29 or 0x28)
//                                   id, address
Adafruit_BNO055 bno = Adafruit_BNO055(55);



const static uint8_t RADIO_ID = 1;
const static uint8_t DESTINATION_RADIO_ID = 0;
const static uint8_t PIN_RADIO_CE = 7;
const static uint8_t PIN_RADIO_CSN = 8;

const static uint8_t RADIO_PACKET_DATA_SIZE = 30;

struct RadioPacket
{
    uint8_t SentenceId;   // Identifier for the long string of data being sent.
    uint8_t PacketNumber; // Locator used to order the packets making up the data, e.g. 0, 1, or 2.
    uint8_t Data[RADIO_PACKET_DATA_SIZE]; // 30 since the radio's max packet size = 32 and we have 2 bytes of metadata.
};




NRFLite _radio;
uint8_t _sentenceId;
uint32_t _lastSendTime;

// Need to pre-declare these functions since they use structs.
bool sendPacket(RadioPacket packet);
void printPacket(RadioPacket packet);



void setup()
{
    Serial.begin(115200);

    Serial.println("Orientation Sensor Test"); Serial.println("");
  Wire.begin();
  
  //channel to open
  tcaselect(0);
  Serial.println("did we make it here");
  /* Initialise the sensor */
  if (!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while (1);
  }

  


  
    if (!_radio.init(RADIO_ID, PIN_RADIO_CE, PIN_RADIO_CSN))
    {
        Serial.println("Cannot communicate with radio");
        while (1); // Wait here forever.
    }
    
    //_radio.printDetails();
    delay(5000);

}

void loop()
{
    Serial.println("---");
    tcaselect(0);
    // Send a sentence once every 0.5 seconds.
    if (millis() - _lastSendTime > 299)
    {
        _lastSendTime = millis();
        uint8_t system, gyro, accel, mg = 0;
        bno.getCalibration(&system, &gyro, &accel, &mg);
        imu::Vector<3> acc =bno.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
        imu::Vector<3> gyr =bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
        imu::Vector<3> mag =bno.getVector(Adafruit_BNO055::VECTOR_MAGNETOMETER);
        imu::Quaternion quat = bno.getQuat();
        /* Display the quat data */
//        Serial.print("qW: ");
//        Serial.print(quat.w(), 4);
//        Serial.print(" qX: ");
//        Serial.print(quat.y(), 4);
//        Serial.print(" qY: ");
//        Serial.print(quat.x(), 4);
//        Serial.print(" qZ: ");
//        Serial.print(quat.z(),4);
//        Serial.println("");
        String quats = String(quat.w())+","+String(quat.x())+","+String(quat.y())+","+String(quat.z());
        String movement = String(acc.x())+","+String(acc.y())+","+String(acc.z())+","+String(gyr.x())+","+String(gyr.y())+","+String(gyr.z())+","+String(mag.x())+","+String(mag.y())+","+String(mag.z());
        String filler = "Some filler stuff";
        String msg = movement+","+quats;
        sendSentence(msg, _sentenceId);
        _sentenceId++;
    }
    
    
   

    delay(300);
}





void sendSentence(String sentence, uint8_t sentenceId)
{
    Serial.println("Sending Sentence " + String(sentenceId));
    Serial.println("  " + sentence);
    Serial.println("  Sentence Length = " + String(sentence.length()));

    RadioPacket packet;
    packet.SentenceId = sentenceId;
    packet.PacketNumber = 0;
    uint8_t packetDataIndex = 0;
    uint8_t sentenceRealLength = sentence.length() + 1; // Need 1 extra byte for the string's null terminator character.

    // Loop through each character in the sentence and add them to the radio packet until the packet is full, send it,
    // then fill the packet with the next set of characters.  We'll continue until all characters, including the null
    // terminator, have been sent.
    for (uint8_t sentenceDataIndex = 0; sentenceDataIndex < sentenceRealLength; sentenceDataIndex++)
    {
        packet.Data[packetDataIndex] = sentence[sentenceDataIndex]; // Copy character from the sentence into the packet.
        bool packetIsFull = packetDataIndex + 1 == RADIO_PACKET_DATA_SIZE;
        bool endOfSentence = sentenceDataIndex + 1 == sentenceRealLength;

        if (packetIsFull || endOfSentence)
        {
            sendPacket(packet);
            packet.PacketNumber++;
            packetDataIndex = 0;
            
        }
        else
        {
            // Increment the location where the next sentence character will be placed within the radio packet.
            packetDataIndex++;
        }
    }
}

bool sendPacket(RadioPacket packet)
{
    Serial.print("  Sending Packet " + String(packet.PacketNumber) + " - ");

    if (_radio.send(DESTINATION_RADIO_ID, &packet, sizeof(packet), NRFLite::NO_ACK))
    {
        printPacket(packet);
        return true;
    }
    else
    {
        Serial.println("Error");
        return false;
    }
}

void printPacket(RadioPacket packet)
{
    // Strings in C are arrays of char characters and they must end with a '\0' null character.
    // In order to print the array of data that's contained in the RadioPacket, we'll need to ensure
    // the data ends with such a character.

    char arrayToPrint[RADIO_PACKET_DATA_SIZE + 1]; // Allow 1 more character in case we need to add a null terminator.
    bool arrayIsTerminated = false;
    uint8_t i = 0;

    while (i < RADIO_PACKET_DATA_SIZE)
    {
        arrayToPrint[i] = packet.Data[i];           // Copy data from the packet into the char array.
        arrayIsTerminated = packet.Data[i] == '\0'; // See if the data is the null terminator character.

        if (arrayIsTerminated)
        {
            break;
        }
        else
        {
            i++; // Prepare to copy the next piece of data from the packet.
        }
    }

    if (!arrayIsTerminated)
    {
        // Add a null terminator so we can print the array as a string.
        arrayToPrint[i] = '\0';
    }

    Serial.println(String(arrayToPrint));
}
