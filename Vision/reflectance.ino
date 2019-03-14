int leftChan[] = {0, 1, 2, 3, 4};
int rightChan[] = {5, 6, 7, 8, 9};
int dataOutLeft = 10;
int dataOutCenter = 11;
int dataOutRight = 12;
int threshold = 0;
int oldTime, newTime, elapsed;
void setup() {
  pinMode(7, OUTPUT);
  pinMode(8, OUTPUT);
}

void loop() {
//    int leftCount = trackLeftChannels(leftChan);
//    int rightCount = trackRightChannels(0);
//    if (leftCount == 1 && rightCount == 1) {
//        digitalWrite(dataOutCenter, HIGH);
//    }
//    else if (leftCount < rightCount) {
//        digitalWrite(dataOutRight, HIGH);
//    }
//    else if (leftCount > rightCount) 
  testSensor(0);
}

//int trackRightChannels(int chans) {
//   // for (int i = 0; i < chans.length; i++) {
//       digitalWrite(i, HIGH);
//       delay(1);
//
//       oldTime = millis();
//       pinMode(i, INPUT);
//
//       while (digitalRead(i) == HIGH) { }
//
//       newTime = millis();
//       elapsed = newTime - oldTime;
//
//       if (elapsed > threshold) {
//           rightCount++;
//       }
//   // }
//   return rightCount;
//}
//
//int trackLeftChannels(int chans[]) {
//    for (int i = 0; i < chans.length; i++) {
//       digitalWrite(i, HIGH);
//       delay(1);
//
//       oldTime = millis();
//       pinMode(i, INPUT);
//
//       while (digitalRead(i) == HIGH) { }
//
//       newTime = millis();
//       elapsed = newTime - oldTime;
//
//       if (elapsed > threshold) {
//           leftCount++;
//       }
//       return leftCount;
//   }
//}

void testSensor(int sensor) {
  digitalWrite(sensor, HIGH);
  delay(1);
  
  int oldTime = millis();
  pinMode(sensor, HIGH);

  while (digitalRead == HIGH) { }

  int newTime = millis();
  int elapsed = newTime - oldTime;

  Serial.println(elapsed);
}
