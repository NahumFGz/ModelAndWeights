from services.predict import predict_glaucoma


model_name = 'VGG16'
img_dir = 'E://Tesis-zSplitKfolds//03. Resultados//zAllDataSet//Fold0//test//normal//Im001_ACRIMA.jpg'

def main():
    print(predict_glaucoma(model_name, img_dir))

if __name__ == "__main__":
    main()
