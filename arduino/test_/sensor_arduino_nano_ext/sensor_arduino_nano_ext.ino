#include <DHT.h>

const int gpio_pin_dht_pin = 13;

DHT dht(gpio_pin_dht_pin, DHT22);

#define READ_TEMPERATURE() dht.readTemperature()
#define READ_HUMIDITY()    dht.readHumidity()

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize the DHT sensor
  dht.begin();

  // Waiting for the peripheral for being ready
  delay(2000);
}

void loop() {
  //  Moved the read sensor values stuff from setup() to here
  Serial.print("Test Temperature = ");
  Serial.print(READ_TEMPERATURE(), 2);
  Serial.println(" °C");
  Serial.print("Test Humidity = ");
  Serial.print(READ_HUMIDITY(), 2);
  Serial.println(" %");

  delay(2000);
}
