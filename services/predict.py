from tools.imagedims import getImageDims
from tools.loadimage import load_image
from tools.loadmodel import load_model, load_weights


def predict_glaucoma(model_name, img_dir):
    #Ruta de modelo
    trained_model_dir = './models/' + model_name + '/'

    #Definir dimensiones
    longitud, altura = getImageDims(model_name)

    #Cargar modelo
    model = load_model(trained_model_dir, model_name)
    #Cargar pesos
    load_weights(model,trained_model_dir,model_name)

    #Cargar Imagen
    img = load_image(img_dir, longitud, altura)

    #Predecir
    pred = model.predict(img)

    return pred[0][0], pred[0][1]


