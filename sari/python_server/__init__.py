from flask import Flask

app = Flask(__name__)

app.secret_key = 'HELLO YOUR GAMBAYOUTAR HAS VIRUS'

from python_server import utils
from python_server import views
