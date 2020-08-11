from keras.models import model_from_json
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os


longitud, altura = 224,224
modelo = 'E:\Tesis-zTestImg\models\VGG16\model.json'
pesos_modelo = 'E:\Tesis-zTestImg\models\VGG16\VGG16_weights.h5'

def load_image(img_path, show=False):


    img = image.load_img(img_path, target_size=(longitud, altura))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor


if __name__ == "__main__":

    # load model
    json_file = open(modelo, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    cnn = model_from_json(loaded_model_json)
    cnn.load_weights(pesos_modelo)

    #cnn.summary()

    # image path
    img_path = 'E://Tesis-zSplitKfolds//03. Resultados//zAllDataSet//Fold0//test//normal//Im001_ACRIMA.jpg'    # dog
    #img_path = '/media/data/dogscats/test1/19.jpg'      # cat

    # load a single image
    new_image = load_image(img_path)

    # check prediction
    pred = cnn.predict(new_image)
    #[[0.99846613 0.0015339 ]]
    '''
    [[0.9289743  0.07102568]]
    92.897%
    7.103%
    '''
    
    print(pred)
    print("{0:.3%}".format(pred[0][0]))
    print("{0:.3%}".format(pred[0][1]))