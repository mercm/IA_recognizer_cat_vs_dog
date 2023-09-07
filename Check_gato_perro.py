import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

class Check():
    def __init__(self, epochs):
        self.epochs = epochs

    def check_prediction(self, y_pred):
        print(y_pred)

    def check_matriz_confusion(self, test_y, y_pred):
        matrix = confusion_matrix(test_y, y_pred)
        
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(x=j, y=i,s=matrix[i, j], va='center', ha='center', size='xx-large')

        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.show()

    def check_metricas(self, test_y, y_pred):
        print(classification_report(test_y, y_pred))

    def check_metricas_plot(self, history):
        acc = history['accuracy']
        val_acc = history['val_accuracy']

        loss = history['loss']
        val_loss = history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(self.epochs, self.epochs))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def check_dataset(self, features, labels):
        print("longitudes: ")
        print(len(features))
        print(len(labels))
        print("tipos: ")
        print(type(features))
        print(type(labels))

    def check_dataset_test(self, features_test, test_labels):
        print("longitudes: ")
        print(len(features_test))
        print(len(test_labels))
        print("tipos: ")
        print(type(features_test))
        print(type(test_labels))

    def check_formato_datos(self, train_features, train_y):
        print(type(train_features))
        print(train_features.shape)
        
        print(type(train_y))
        print(train_y.shape)

    def check_formato_datos_test(self, test_x, test_labels):    
        print(type(test_x))
        print(test_x.shape)

        print(type(test_labels))
        print(test_labels.shape)

    def check_prep_modelo(self, train_x, valid_x):
        print(train_x.shape)
        print(valid_x.shape)
        print(type(valid_x))
        print(type(train_x))

    def check_model(self, model):
        print(model.summary())

    def check_perdida(self, history):
        plt.xlabel("# Epoca")
        plt.ylabel("Magnitud de pérdida")
        plt.plot(history["loss"])

    def check_plot_labels(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        plt.bar(['cat', 'dog'], counts)
        plt.show()