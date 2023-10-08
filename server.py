from flask import Flask, Response, request
from functools import wraps

import cv2
import configparser

app = Flask(__name__)

config = configparser.ConfigParser()
config.read('config.cfg')

remote_http_auth = config.getboolean('settings', 'remote_http_auth')
remote_http_username = config.get('settings', 'remote_http_username')
remote_http_password = config.get('settings', 'remote_http_password')

def check_auth(username, password):
    return username == remote_http_username and password == remote_http_password

def authenticate():
    return Response('Could not verify your access level for that URL.\n'
                    'You have to login with proper credentials', 401,
                    {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(function):
    """Decorer to assure required credentials"""
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return function(*args, **kwargs)
    return decorated


def conditional_requires_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if remote_http_auth:
            auth = request.authorization
            if not auth or not check_auth(auth.username, auth.password):
                return authenticate()
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@conditional_requires_auth
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(host='localhost', port=9990)
