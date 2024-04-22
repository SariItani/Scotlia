from flask import render_template
from python_server import app

@app.route('/')
def hello_world():
    return render_template('index.html')