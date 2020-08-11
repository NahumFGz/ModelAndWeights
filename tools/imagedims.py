def getImageDims(model):
    if model == 'VGG16' or model == 'VGG19':
        longitud, altura = 224, 224

    elif model == 'InceptionV3':
        longitud, altura = 299, 299    

    return longitud, altura