import RPIO as GPIO
from datetime import datetime

lower_time = 0
upper_time = 0
chan_list = [11, 12, 13, 15, 16, 18, 22, 7, 3, 5, 24, 26]
left_chan = [11, 12, 13, 15, 16, 18]
right_chan = [22, 7, 3, 5, 24, 26]
spacing = 5

GPIO.setmode(GPIO.BOARD)
GPIO.setup(chan_list, GPIO.OUT, initial=GPIO.HIGH)


def track_sensors(channels):
    activated = []
    for channel in channels:
        old_time = datetime.now().microsecond
        GPIO.setup(channel, GPIO.IN)
        while GPIO.input(channel) == GPIO.HIGH:
            datetime.delay(0.001)
        new_time = datetime.now().microsecond
        elapsed = new_time - old_time
        if elapsed < upper_time and elapsed > lower_time:
            activated.append(channel)
        
    return activated


def correct_alignment():
    leftActivated = track_sensors(left_chan)
    rightActivated = track_sensors(right_chan)
    if len(leftActivated) == 1 and len(rightActivated) == 1:
        return 0
    elif len(leftActivated) > len(rightActivated):
        return (len(leftActivated) * -spacing) / 100
    elif len(rightActivated) > len(leftActivated):
        return (len(rightActivated) * spacing) / 100

