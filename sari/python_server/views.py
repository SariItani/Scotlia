from flask import render_template
from python_server import app, utils
import cv2
import matplotlib.pyplot as plt

@app.route('/')
def hello_world():
    fire = utils.map_with_colors("./python_server/static/fire.jpg", cell_size=25)
    cv2.imwrite('python_server/static/image.png', fire) # replace this with map

    return render_template('index.html', image='image.png')
