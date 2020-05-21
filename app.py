import os
from flask import Flask, request, jsonify, url_for

from configs.server import MODEL_PATH, LOCAL_HOST, PUBLIC_HOST, WEB_SERVER_PORT

from routes.manage_data import manage_data
from routes.manage_model import manage_model
from routes.predict import predict
from routes.train import train

app = Flask(__name__)
app.register_blueprint(manage_data)
app.register_blueprint(manage_model)
app.register_blueprint(predict)
app.register_blueprint(train)

if __name__ == "__main__":
    app.run(debug=True, host=LOCAL_HOST)

