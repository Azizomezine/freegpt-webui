import secrets

from server.bp import bp
from server.website import Website
from server.backend import Backend_Api
from server.babel import create_babel
from json import load
from flask import Flask, render_template,request
import requests
import numpy as np
import cv2
API_URL = "https://api-inference.huggingface.co/models/microsoft/trocr-large-handwritten"
headers = {"Authorization": "Bearer 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"}
app = Flask(__name__)
@app.route("/ocr", methods=["POST"])
def ocr():
    num_lines = int(request.form["num_lines"])
    upper=0
    lower=110
    T=[]
    result=[]
    response=[]
    generated_text = ""
    cv_image = request.files["image"]
    image_data = cv_image.read()
    nparr = np.fromstring(image_data, np.uint8)
    cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width, _ = cv_image.shape
    for i in range(num_lines):
        T.append(cv_image[upper:lower, 10:width])
        cv2.imwrite("image_recadree"+str(i)+".jpg", T[i])
        upper+=110
        lower+=110
        _, img_bytes = cv2.imencode(".jpg", T[i])
        response_i = requests.post(API_URL, headers=headers, data=img_bytes.tobytes())
        result_i = response_i.json()
        result.append(result_i)
        if(i>0):
          generated_text+=" "
        generated_text += result_i[0]['generated_text']
    print("Generated text: ", generated_text)
    return render_template("client/index.html", generated_text=generated_text)

if __name__ == '__main__':

    # Load configuration from config.json
    config = load(open('config.json', 'r'))
    site_config = config['site_config']
    url_prefix = config.pop('url_prefix')

    # Create the app
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(16)

    # Set up Babel
    create_babel(app)

    # Set up the website routes
    site = Website(bp, url_prefix)
    for route in site.routes:
        bp.add_url_rule(
            route,
            view_func=site.routes[route]['function'],
            methods=site.routes[route]['methods'],
        )

    # Set up the backend API routes
    backend_api = Backend_Api(bp, config)
    for route in backend_api.routes:
        bp.add_url_rule(
            route,
            view_func=backend_api.routes[route]['function'],
            methods=backend_api.routes[route]['methods'],
        )

    # Register the blueprint
    app.register_blueprint(bp, url_prefix=url_prefix)

    # Run the Flask server
    print(f"Running on {site_config['port']}{url_prefix}")
    app.run(**site_config)
    print(f"Closing port {site_config['port']}")
