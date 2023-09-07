import zipfile
import os
import pandas as pd
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import tensorflow as tf
import numpy as np
import pickle

from Check_practica import Check

def unzip():
    # code from : https://www.kaggle.com/code/nirmit19/cats-vs-dogs-vgg-19

    with zipfile.ZipFile(TRAIN_DIR_UNZIP, 'r') as zipp:
        zipp.extractall(INI_FILES)

def unzip_test():
    # code from : https://www.kaggle.com/code/nirmit19/cats-vs-dogs-vgg-19

    with zipfile.ZipFile(TEST_DIR_UNZIP, 'r') as zipp:
        zipp.extractall(INI_FILES)

def guardar_datos(features, labels):
    np.save(FEATURES_PATH, features)
    np.save(LABELS_PATH, labels)

def guardar_datos_test(features_test, labels_test):
    np.save(FEATURES_TEST_PATH, features_test)
    np.save(LABELS_TEST_PATH, labels_test)

def cargar_datos():
    features = np.load(FEATURES_PATH, allow_pickle=True)
    labels = np.load(LABELS_PATH, allow_pickle=True)

    return features, labels

def cargar_datos_test():
    features_test = np.load(FEATURES_TEST_PATH, allow_pickle=True)
    test_labels = np.load(LABELS_TEST_PATH, allow_pickle=True)

    return features_test, test_labels

def guardar_modelo(history, model):
    model.save_weights(CHECKPOINT_MODEL_PATH)

    with open(CHECKPOINT_HISTORY_PATH, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

def cargar_modelo(model):
    model.load_weights(CHECKPOINT_MODEL_PATH)

    with open(CHECKPOINT_HISTORY_PATH, "rb") as file_pi:
        history = pickle.load(file_pi)

    return model, history

def etiquetar_imagenes(train_images, train_dir, features, labels):
    #Procesamiento de imágenes
    for image in tqdm(train_images, desc="Processing Train Images"):
        if "cat" in image:
            label = 0
        else :
            label = 1

        image_dir=train_dir+"/"+image
        image = cv2.imread(image_dir)
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        #image_array = np.array(image_resized)
        image = image.astype('float32')
        image = image / 255.0
        #image = image.flatten()
        features.append(image)
        labels.append(label)
    return features, labels

def etiquetar_imagenes_test(test_images, test_dir, labels):
    #Procesamiento de imágenes de test
    test_labels = [0 for element in range(labels.shape[0])]
    features_test = []
    i=0
    for image in tqdm(test_images, desc="Processing Test Images"):
        #etiquetas
        id = int(image.split('.jpg')[0])
        test_labels[i] = labels['labels'][id-1]
        i=i+1

        #imagenes
        image_dir=test_dir+"/"+image
        image = cv2.imread(image_dir)
        image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        #image_array = np.array(image_resized)
        image = image.astype('float32')
        image = image / 255.0
        #image = image.flatten()
        features_test.append(image)
    return features_test, test_labels

def generar_datos(ck):
    #CREAR DATASET
    train_images = os.listdir(TRAIN_DIR)
    features = []
    labels = []

    features, labels = etiquetar_imagenes(train_images, TRAIN_DIR, features, labels)
    ck.check_dataset(features, labels)

    guardar_datos(features, labels)

    return features, labels

def generar_datos_test():
    #CREAR DATASET TEST
    features_test = []

    #if DATOS_TEST_PARCIAL:
    labels = pd.read_csv('Unidad 2. Deep Learning/dogs-vs-cats/etiquetas_test_828.csv', sep=';')
    test_images = os.listdir(TEST_DIR_PARCIAL)
    '''else:
        #No utilizar
        labels = []
        test_images = os.listdir(TEST_DIR)'''

    features_test, test_labels = etiquetar_imagenes_test(test_images, TEST_DIR, labels)

    #ck.check_dataset_test(features_test, test_labels)

    guardar_datos_test(features_test, test_labels)

    return features_test, test_labels

def formatear_datos(features, labels):
    #train
    train_features = np.array(features)
    train_y = np.array(labels)

    return train_features, train_y

def formatear_datos_test(features_test, test_labels):
    #test
    test_x = np.array(features_test)
    test_labels = np.array(test_labels)

    return test_x, test_labels

def predict(model, test_x):
    y_pred = model.predict(test_x)
    y_pred = y_pred.flatten()
    y_pred = np.where(y_pred > 0.5, 1, 0)

    return y_pred

def evaluar(model, test_x, test_labels):
    test_eval = model.evaluate(test_x, test_labels, verbose=1)
    print('Test loss: ', test_eval[0])
    print('Test accuracy: ', test_eval[1])

def test(model, ck):
    
    if GENERAR_DATOS_TEST:
        unzip_test()
        features_test, test_labels = generar_datos_test()
    else:
        features_test, test_labels = cargar_datos_test()

    #ck.check_plot_labels(test_labels)

    test_features, test_labels = formatear_datos_test(features_test, test_labels)
    #ck.check_formato_datos_test(test_features, test_labels)
    
    evaluar(model, test_features, test_labels)

    #Prediccion
    y_pred = predict(model, test_features)
    ck.check_prediction(y_pred)
    ck.check_matriz_confusion(test_labels, y_pred)
    ck.check_metricas(test_labels, y_pred)

def modelo():
    #DEFINIR MODELO
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.30))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.50))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.50))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(NUM_CLASES, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer= tf.keras.optimizers.Adam(), metrics=['accuracy'])

    return model

def callbacks():
    #AÑADIR PARA OPTIMIZACION (EVITAR VUELTAS DE MÁS)
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    earlystop = EarlyStopping(patience=PATIENCE)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    callbacks = [earlystop, learning_rate_reduction]

    return callbacks

def entrenamiento(train_x, valid_x, train_label, valid_label, model, callbacks):
    #ENTRENAMIENTO
    history = model.fit(
        train_x,
        train_label,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
        validation_data=(valid_x, valid_label),
        callbacks=callbacks
    )

    return history, model

def generar_modelo(ck):
    if GENERAR_DATOS:
        #Se descomprimen y etiquetan los datos desde los archivos
        unzip()
        features, labels = generar_datos(ck)
    else:
        features, labels = cargar_datos()

    #ck.check_plot_labels(labels)

    train_features, train_y = formatear_datos(features, labels)
    #ck.check_formato_datos(train_features, train_y)

    #Preparacion para modelo
    train_x, valid_x, train_label, valid_label = train_test_split(train_features, train_y, test_size=0.20, random_state=42)
    #ck.check_prep_modelo(train_x, valid_x, train_label, valid_label)

    #Definir modelo
    model = modelo()
    ck.check_model(model)

    #Optimizacion
    cb = callbacks()
    #cb = None

    #Entrenamiento
    history, model = entrenamiento(train_x, valid_x, train_label, valid_label, model, cb)

    guardar_modelo(history, model)

    return history.history, model

def recuperar_modelo(ck):
    #Definir modelo
    model = modelo()

    model, history = cargar_modelo(model)
    ck.check_model(model)

    return model, history

def main():
    ck = Check(EPOCHS)

    if GENERAR_MODELO:
        #Genera el modelo de 0
        history, model = generar_modelo(ck)
    else:
        #Carga un modelo previamente generado y guardado
        model, history = recuperar_modelo(ck)

    #ck.check_perdida(history)
    ck.check_metricas_plot(history)

    test(model, ck)

    print("fin")
    

IMAGE_WIDTH = 50 
IMAGE_HEIGHT = 50 
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

BATCH_SIZE = 50
NUM_CLASES = 1
EPOCHS = 55
PATIENCE = 40

GENERAR_MODELO = False
GENERAR_DATOS = False
GENERAR_DATOS_TEST = False
#MODEL_NUM = 5
#SAVE_COUNT = 4

INI_FILES = "Unidad 2. Deep Learning/dogs-vs-cats"
TRAIN_DIR_UNZIP = "Unidad 2. Deep Learning/dogs-vs-cats/train.zip"
TEST_DIR_UNZIP = "Unidad 2. Deep Learning/dogs-vs-cats/test1.zip"

TRAIN_DIR = "Unidad 2. Deep Learning/dogs-vs-cats/train"
TEST_DIR = "Unidad 2. Deep Learning/dogs-vs-cats/test"
TEST_DIR_PARCIAL = "Unidad 2. Deep Learning/dogs-vs-cats/test_parcial"

CHECKPOINT_MODEL_PATH = "Unidad 2. Deep Learning/checkpoint/model_1.h5" #"_" + str(MODEL_NUM) + "_" + str(SAVE_COUNT) + ".h5"
CHECKPOINT_HISTORY_PATH = "Unidad 2. Deep Learning/checkpoint/history_1.npy" #"_" + str(MODEL_NUM) + "_" + str(SAVE_COUNT) + ".npy"

FEATURES_PATH = "Unidad 2. Deep Learning/checkpoint/features.npy"
LABELS_PATH = "Unidad 2. Deep Learning/checkpoint/labels.npy"
FEATURES_TEST_PATH = "Unidad 2. Deep Learning/checkpoint/features_test_828.npy"
LABELS_TEST_PATH = "Unidad 2. Deep Learning/checkpoint/labels_test_828.npy"

main()