import os

os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ['TF_USE_CUDNN_BATCHNORM'] = '0'

import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input
from tensorflow.keras.layers import Flatten, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import numpy as np




import math

# training parameters
batch_size = 32 

epochs=201

data_augmentation = True
num_classes = 7

# Para centrar los valores de los pixel y mejorar accuracy
subtract_pixel_mean = True

version = 2

n = 17

if version == 1:
    depth = n * 6 + 2 #104
elif version == 2:
    depth = n * 9 + 2 #155

# Nombre del modelo, profundidad y version
model_type = 'ResNet%dv%d' % (depth, version)


def extraer_imagenes(directorio,tamanio): # Directorio con los datos
    """ La función recibe el directorio que tiene los datos. Las imagenes son guardadas en la lista "imagenes" y la clasificación
    en la lista "clases". Cuando se recorre una carpeta con imágenes registrará en la lista "clases" un número de clasificación
    como tantos elementos hay en la carpeta. Normaliza las imágenes y devuelve las matrices listas para jugar.
    
    """
    clases = []
    imagenes = []
    lista_carpetas = os.listdir(directorio)
    # Recorrer
    k = 0
    for carpeta_clase in lista_carpetas:
        for foto in os.listdir(directorio+'/'+carpeta_clase):
            imagen = image.load_img(directorio+'/'+carpeta_clase+'/'+foto, color_mode='grayscale',target_size=tamanio)  # Cada imagen guardada
            imagenes.append(np.array(imagen))       # Registra el número de carpeta al que pertence
            clases.append(k)
        k+=1
    # Conversión a numpy
    X_train = np.array(imagenes, dtype=np.float32)
    # print(X_train.shape)
    clases = np.array(clases)

    # Formato
    image_size = X_train.shape[1]
    canales= tamanio[2]
    X_train = np.reshape(X_train, [-1, image_size, image_size, canales])
    # print(X_train.shape)
    
    # Normalizar
    X_train = X_train / 63
    # print(X_train.shape)
    y_train = to_categorical(clases)
    return X_train, y_train

path_1 = '/home/cursos/ima543_2025_1/ima543_share/Datasets/FER/train' 
path_2 = '/home/cursos/ima543_2025_1/ima543_share/Datasets/FER/test' 

x_train, y_train = extraer_imagenes(path_1,tamanio=(64,64,1))
x_test, y_test = extraer_imagenes(path_2,tamanio=(64,64,1))

# dimensiones de la imagen input
input_shape = x_train.shape[1:]

print(x_test.shape)
print(y_test.shape)

# Si se solicita sustraer la media de los pixels
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convertir vectores clases a matrices binarias
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#Hacemos flatten para que las dimensiones calcen
y_train = np.array([row[0] for row in y_train])
y_test = np.array([row[0] for row in y_test])

def lr_schedule(epoch):
    """Ajuste del Learning Rate
    Learning rate se reduce después de 80, 120, 160, 180 epocas.
    Esta función se llama automaticamente después de cada época durante
    el entrenamiento como parte de los callbacks.
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,activation='relu',batch_normalization=True,conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    Arguments:
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    Returns:
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size = kernel_size,
                  strides     = strides,
                  padding     = 'same',
                  kernel_initializer = 'he_normal',
                  kernel_regularizer = l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v2(input_shape, depth, num_classes=7):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or 
    also known as bottleneck layer.
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, 
    the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, 
    while the number of filter maps is
    doubled. Within each stage, the layers have 
    the same number filters and the same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    Arguments:
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    Returns:
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 110 in [b])')
    # start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    
    # v2 performs Conv2D with BN-ReLU
    # on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,num_filters=num_filters_in,conv_first=True)

    # instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                # first layer and first stage
                if res_block == 0:  
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                # first layer but not first stage
                if res_block == 0:
                    # downsample
                    strides = 2 

            # bottleneck residual unit
            y = resnet_layer(inputs=x,num_filters=num_filters_in,kernel_size=1,strides=strides,
                             activation=activation,batch_normalization=batch_normalization,conv_first=False)
            y = resnet_layer(inputs=y,num_filters=num_filters_in,conv_first=False)
            y = resnet_layer(inputs=y,num_filters=num_filters_out,kernel_size=1,conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection
                # to match changed dims
                x = resnet_layer(inputs=x,num_filters=num_filters_out,kernel_size=1,
                                 strides=strides,activation=None,batch_normalization=False)
            x = add([x, y])

        num_filters_in = num_filters_out

    # add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,activation='softmax',kernel_initializer='he_normal')(y)

    # instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=lr_schedule(0)),metrics=['acc'])
model.summary()

# prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'taller1_resnet_v2_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,monitor='val_acc',verbose=2,save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,
              validation_data=(x_test, y_test),shuffle=True,callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # this will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    steps_per_epoch =  math.ceil(len(x_train) / batch_size)
    # fit the model on the batches generated by datagen.flow().

    print("Chequeo de dimensiones")
    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)
    print("y_train[0]:", y_train[0])


    history = model.fit(x=datagen.flow(x_train, y_train, batch_size=batch_size),
              verbose=2,epochs=epochs,validation_data=(x_test, y_test),
              steps_per_epoch=steps_per_epoch,callbacks=callbacks)


# score trained model
scores = model.evaluate(x_test,y_test,batch_size=batch_size,verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

plt.plot(history.history['acc']) 
plt.plot(history.history['val_acc'])
plt.suptitle('Exactitud del modelo')
plt.title('Modelo Resnet v2')
plt.ylabel('Exactitud')
plt.xlabel('Épocas')
plt.legend(['Entrenamiento', 'Test'], loc='best') 
plt.savefig('accuracy_densenet.png')
plt.close() 

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.suptitle('Función de Pérdidas')
plt.title('Modelo Resnet v2')
plt.ylabel('Valor pérdida')
plt.xlabel('Épocas')
plt.legend(['Entrenamiento', 'Test'], loc='best')
plt.savefig('loss_densenet.png')
plt.close()