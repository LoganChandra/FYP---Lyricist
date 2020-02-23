import numpy as np
import keras.models
from keras.models import model_from_json
import tensorflow as tf

def init():
    json_file = open('/model/Lyric Generation/json_h5/Indie_1000songs_5sl_100_100_100_vsize.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into model
    loaded_model.load_weights('/model/Lyric Generation/json_h5/Indie_1000songs_5sl_100_100_100_vsize.h5')
    print("Loaded Model from disk")

    # Compile and evaluate loaded model
    loaded_model.Compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    return loaded_model