from flask import render_template
from python_server import app, utils

@app.route('/')
def hello_world():
    # call function, get image
    image = utils.process_image('forset.jpeg')
    return render_template('index.html', image=image)
