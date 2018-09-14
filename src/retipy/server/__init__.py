from flask import Flask
app = Flask(__name__)
base_url = "/retipy/"
import retipy.server.endpoint_main
import retipy.server.endpoint_tortuosity
import retipy.server.endpoint_landmarks
