# Source: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html

# import the necessary packages
import requests

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5005/predict"
IMAGE_PATH = "TestImages/maxxi.jpg"

classes = ['Tarantula', 'Widow Spider', 'Long-Jawed Orb Weaver', 'Jumping Spider', 'Nursery Web Spider',
           'Cellar Spider', 'Orb-Weaver Spider', 'Lynx Spider', 'Huntsman Spider', 'Tangle-Web Spider',
           'Crab Spider']
actual_classes = ['Tarantula', '', '', 'Jumping Spider', 'Nursery Web Spider',
           '', 'Orb-Weaver Spider', '', '', 'Tangle-Web Spider',
           '']

# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# ensure the request was successful
if r["success"]:
    # loop over the predictions and display them
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, result["species"],
            result["probability"]))

# otherwise, the request failed
else:
    print("Request failed")
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(i + 1, result["species"],
            result["probability"]))