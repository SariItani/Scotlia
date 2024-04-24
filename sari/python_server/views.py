from flask import render_template
from python_server import app, utils
import serial
import time
from utils import detect_stuff

@app.route('/')
def hello_world():
    image = utils.process_image('forset.jpeg')
    images = detect_stuff()
    random_image = image[0][1]

    ser = serial.Serial('/dev/ttyACM0',9600, timeout=1)
    ser.reset_input_buffer()

    state0 = ser.readline().decode('utf-8').rstrip()
    print(state0)
    time.sleep(1)

    line = ser.readline().decode('utf-8').rstrip()
    while True:
        print(line)
        if(line == state0):
            break
            # time.sleep(1)
            # return render_template('index.html', image=image, message=line)

        else:
            line = ser.readline().decode('utf-8').rstrip()
            time.sleep(1)
            return render_template('index.html', image=random_image, message=line)

    return render_template('index.html', image=random_image, message=line)
