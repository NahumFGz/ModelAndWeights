import os
from keras.models import model_from_json

def load_model(trained_model_dir,model_name):
    model_name = 'model.json'
    json_file = open(os.path.join(trained_model_dir, model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    return model_from_json(loaded_model_json)

def load_weights(model,trained_model_dir,model_name):
    model_name += '_weights.h5'
    model.load_weights(os.path.join(trained_model_dir, model_name))



