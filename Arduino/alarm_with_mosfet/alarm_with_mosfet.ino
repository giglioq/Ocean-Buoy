// Arduino DS3232RTC Library
// https://github.com/JChristensen/DS3232RTC
// Copyright (C) 2018 by Jack Christensen and licensed under
// GNU GPL v3.0, https://www.gnu.org/licenses/gpl.html
//
// DS3231/DS3232 Alarm Example Sketch #1
//
// Set Alarm 1 to occur once a minute at 5 seconds after the minute.
// Detect the alarm by polling the RTC alarm flag.
//
// Hardware:
// Arduino Uno, DS3231 RTC.
// Connect RTC SDA to Arduino pin A4.
// Connect RTC SCL to Arduino pin A5.
//
// Jack Christensen 16Sep2017

#include <DS3232RTC.h>      // https://github.com/JChristensen/DS3232RTC
#include <Streaming.h>      // http://arduiniana.org/libraries/streaming/



void setup()
{
    Serial.begin(115200);

    // initialize the alarms to known values, clear the alarm flags, clear the alarm interrupt flags
    RTC.setAlarm(ALM1_MATCH_DATE, 0, 0, 0, 1);
    RTC.setAlarm(ALM2_MATCH_DATE, 0, 0, 0, 1);
    RTC.alarm(ALARM_1);
    RTC.alarm(ALARM_2);
    RTC.alarmInterrupt(ALARM_1, false);
    RTC.alarmInterrupt(ALARM_2, false);
    RTC.squareWave(SQWAVE_NONE);

    // set the RTC time and date to the compile time
    RTC.set(compileTime());

    
}

void loop()
{
    turn_off();// no need to bombard the RTC continuously
    delay(2000);
    
    set_new_alarm();
    delay(7000);
}


void set_new_alarm(){
  int wait_time = 5;
  RTC.alarmInterrupt(ALARM_1, true);  //pin high
  RTC.squareWave(SQWAVE_NONE); // dont change alarm pin
  setTime(0, 0, 0, 1, 1, 1970);
  RTC.set(now());
  
  // set new alarm
  RTC.setAlarm(ALM1_MATCH_SECONDS, 0, wait_time, 0, 1);
  Serial.println("alarm set");
}

void turn_off(){
  // clear old alarm flag - turning off system
  RTC.alarm(ALARM_1);
}




void printDateTime(time_t t)
{
    Serial << ((day(t)<10) ? "0" : "") << _DEC(day(t));
    Serial << monthShortStr(month(t)) << _DEC(year(t)) << ' ';
    Serial << ((hour(t)<10) ? "0" : "") << _DEC(hour(t)) << ':';
    Serial << ((minute(t)<10) ? "0" : "") << _DEC(minute(t)) << ':';
    Serial << ((second(t)<10) ? "0" : "") << _DEC(second(t));
}

// function to return the compile date and time as a time_t value
time_t compileTime()
{
    const time_t FUDGE(10);    //fudge factor to allow for upload time, etc. (seconds, YMMV)
    const char *compDate = __DATE__, *compTime = __TIME__, *months = "JanFebMarAprMayJunJulAugSepOctNovDec";
    char compMon[4], *m;

    strncpy(compMon, compDate, 3);
    compMon[3] = '\0';
    m = strstr(months, compMon);

    tmElements_t tm;
    tm.Month = ((m - months) / 3 + 1);
    tm.Day = atoi(compDate + 4);
    tm.Year = atoi(compDate + 7) - 1970;
    tm.Hour = atoi(compTime);
    tm.Minute = atoi(compTime + 3);
    tm.Second = atoi(compTime + 6);

    time_t t = makeTime(tm);
    return t + FUDGE;        //add fudge factor to allow for compile time
}
