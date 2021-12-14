# Source: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

from tensorflow import keras
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

import os
import sys
import tensorflow as tf

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
num_classes = 11
image_shape = (299, 299)
port = 5005
classes = ['Tarantula', 'Widow Spider', 'Long-Jawed Orb Weaver', 'Jumping Spider', 'Nursery Web Spider',
           'Cellar Spider', 'Orb-Weaver Spider', 'Lynx Spider', 'Huntsman Spider', 'Tangle-Web Spider',
           'Crab Spider']


#
def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model

    model = keras.models.load_model("model_family.h5")


#
def prepare_image(image, target):
    # image is the image
    # target is the target dimensions of the image

    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    # TODO: make sure this is the right format to make predictions off of for the model
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    print("Recieved request")

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=image_shape)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            print(classes)
            print(preds)
            # [Class1, geometricus, Class3, duellica, Class5]
            # results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []
            data["success"] = False

            count = 0
            # add labels to each probability
            for x in classes:
                data["predictions"].append({"species": x, "probability": float(preds[0][count])})
                if float(preds[0][count]) > .65:
                    data["success"] = True
                count = count + 1

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(host="0.0.0.0", port=port)
