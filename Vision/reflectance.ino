int sensorPin = 7;
int dataOutPin = 8;
int threshold = 0;
bool reading = false;
int oldTime, newTime, elapsed;
void setup() {
  pinMode(7, OUTPUT);
  pinMode(8, OUTPUT);
}

void loop() {
  digitalWrite(sensorPin, HIGH);
  delay(1);
  
  oldTime = millis();
  pinMode(7, INPUT);
  
  while (digitalRead(7) == HIGH) {
    
  }
  
  newTime = millis();
  elapsed = newTime - oldTime;
  
  if (elapsed > threshold) {
    digitalWrite(dataOutPin, HIGH);
  }
  else {
    digitalWrite(dataOutPin, LOW);
  }
}
