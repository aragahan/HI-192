from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPool2D


cnn3 = Sequential()
cnn3.add(Conv2D(8,kernel_size=(3,3), activation='relu', input_shape=(256,256,3)))
cnn3.add(MaxPool2D((2, 2)))
cnn3.add(Dropout(0.2))

cnn3.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
cnn3.add(MaxPool2D(pool_size=(2,2)))
cnn3.add(Dropout(0.2))

cnn3.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
cnn3.add(MaxPool2D(pool_size=(2,2)))
cnn3.add(Dropout(0.2))

cnn3.add(Flatten())
cnn3.add(Dropout(0.2))
cnn3.add(Dense(32,activation='relu'))
cnn3.add(Dropout(0.2))
cnn3.add(Dense(1,activation='sigmoid'))

cnn3.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

image_generatorFullAugment = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.15,
    shear_range=0.3,
    zoom_range=0.30,
    samplewise_center=True,
    horizontal_flip=True,
    samplewise_std_normalization=True
)

image_generatorNoAugment = ImageDataGenerator(
    samplewise_std_normalization=True
)


train = image_generatorFullAugment.flow_from_directory('datasets/train/',
                                                       batch_size=32,
                                                       shuffle=True,
                                                       class_mode='binary',
                                                       target_size=(256,256))

validation = image_generatorFullAugment.flow_from_directory('datasets/validation/',
                                                       batch_size=1,
                                                       shuffle=True,
                                                       class_mode='binary',
                                                       target_size=(256,256))

test = image_generatorNoAugment.flow_from_directory('datasets/test/',
                                                       batch_size=1,
                                                       shuffle=False,
                                                       class_mode='binary',
                                                       target_size=(256,256))

r100 = cnn3.fit(train,
                epochs = 1,
                validation_data=validation,
#                class_weight=class_weight,
                batch_size=64,
                verbose=1)


import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(r100.history['loss'],label = 'Loss')
plt.plot(r100.history['val_loss'],label='Val_Loss')
plt.legend()
plt.title('Loss Evolution')


plt.subplot(2,2,2)
plt.plot(r100.history['accuracy'],label='Accuracy')
plt.plot(r100.history['val_accuracy'],label='Val_Accuracy')
plt.legend()
plt.title('Accuracy Evolution')

plt.show()
plt.close()


#VGG19
from keras.applications import VGG19,ResNet50,InceptionV3
#3 layers stay the same
cnn3 = Sequential()
vgg19 = VGG19(include_top=False, input_shape=(256,256,3))
cnn3.add(vgg19)
cnn3.add(Conv2D(16,kernel_size=(3,3), activation='relu', input_shape=(256,256,3),padding='same'))
cnn3.add(MaxPool2D((2, 2),padding='same'))
cnn3.add(Dropout(0.2))

cnn3.add(Conv2D(32, kernel_size=(3,3), activation='relu',padding='same'))
cnn3.add(MaxPool2D(pool_size=(2,2),padding='same'))
cnn3.add(Dropout(0.2))

cnn3.add(Conv2D(64, kernel_size=(3,3), activation='relu',padding='same'))
cnn3.add(MaxPool2D(pool_size=(2,2),padding='same'))
cnn3.add(Dropout(0.2))

cnn3.add(Flatten())
cnn3.add(Dropout(0.2))
cnn3.add(Dense(32,activation='relu'))
cnn3.add(Dropout(0.2))
cnn3.add(Dense(1,activation='sigmoid'))

cnn3.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

image_generatorFullAugment = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.15,
    shear_range=0.3,
    zoom_range=0.30,
    samplewise_center=True,
    horizontal_flip=True,
    samplewise_std_normalization=True
)

image_generatorNoAugment = ImageDataGenerator(
    samplewise_std_normalization=True
)


train = image_generatorFullAugment.flow_from_directory('datasets/train/',
                                                       batch_size=32,
                                                       shuffle=True,
                                                       class_mode='binary',
                                                       target_size=(256,256))

validation = image_generatorFullAugment.flow_from_directory('datasets/validation/',
                                                       batch_size=1,
                                                       shuffle=True,
                                                       class_mode='binary',
                                                       target_size=(256,256))

test = image_generatorNoAugment.flow_from_directory('datasets/test/',
                                                       batch_size=1,
                                                       shuffle=False,
                                                       class_mode='binary',
                                                       target_size=(256,256))

r100 = cnn3.fit(train,
                epochs = 1,
                validation_data=validation,
#                class_weight=class_weight,
                batch_size=64,
                verbose=1)


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(r100.history['loss'],label = 'Loss')
plt.plot(r100.history['val_loss'],label='Val_Loss')
plt.legend()
plt.title('Loss Evolution')


plt.subplot(2,2,2)
plt.plot(r100.history['accuracy'],label='Accuracy')
plt.plot(r100.history['val_accuracy'],label='Val_Accuracy')
plt.legend()
plt.title('Accuracy Evolution')

plt.show()
plt.close()

#Resnet50

cnn3 = Sequential()
resnet50 = ResNet50(include_top=False, input_shape=(256,256,3))
cnn3.add(resnet50)
cnn3.add(Conv2D(16,kernel_size=(3,3), activation='relu', input_shape=(256,256,3),padding='same'))
cnn3.add(MaxPool2D((2, 2),padding='same'))
cnn3.add(Dropout(0.2))

cnn3.add(Conv2D(32, kernel_size=(3,3), activation='relu',padding='same'))
cnn3.add(MaxPool2D(pool_size=(2,2),padding='same'))
cnn3.add(Dropout(0.2))

cnn3.add(Conv2D(64, kernel_size=(3,3), activation='relu',padding='same'))
cnn3.add(MaxPool2D(pool_size=(2,2),padding='same'))
cnn3.add(Dropout(0.2))

cnn3.add(Flatten())
cnn3.add(Dropout(0.2))
cnn3.add(Dense(32,activation='relu'))
cnn3.add(Dropout(0.2))
cnn3.add(Dense(1,activation='sigmoid'))

cnn3.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

image_generatorFullAugment = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.15,
    shear_range=0.3,
    zoom_range=0.30,
    samplewise_center=True,
    horizontal_flip=True,
    samplewise_std_normalization=True
)

image_generatorNoAugment = ImageDataGenerator(
    samplewise_std_normalization=True
)


train = image_generatorFullAugment.flow_from_directory('datasets/train/',
                                                       batch_size=32,
                                                       shuffle=True,
                                                       class_mode='binary',
                                                       target_size=(256,256))

validation = image_generatorFullAugment.flow_from_directory('datasets/validation/',
                                                       batch_size=1,
                                                       shuffle=True,
                                                       class_mode='binary',
                                                       target_size=(256,256))

test = image_generatorNoAugment.flow_from_directory('datasets/test/',
                                                       batch_size=1,
                                                       shuffle=False,
                                                       class_mode='binary',
                                                       target_size=(256,256))

r100 = cnn3.fit(train,
                epochs = 1,
                validation_data=validation,
#                class_weight=class_weight,
                batch_size=64,
                verbose=1)


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(r100.history['loss'],label = 'Loss')
plt.plot(r100.history['val_loss'],label='Val_Loss')
plt.legend()
plt.title('Loss Evolution')


plt.subplot(2,2,2)
plt.plot(r100.history['accuracy'],label='Accuracy')
plt.plot(r100.history['val_accuracy'],label='Val_Accuracy')
plt.legend()
plt.title('Accuracy Evolution')

plt.show()
plt.close()

#InceptionV3

cnn3 = Sequential()
inceptionv3 = InceptionV3(include_top=False, input_shape=(256,256,3))
cnn3.add(inceptionv3)
cnn3.add(Conv2D(16,kernel_size=(3,3), activation='relu', input_shape=(256,256,3),padding='same'))
cnn3.add(MaxPool2D((2, 2),padding='same'))
cnn3.add(Dropout(0.2))

cnn3.add(Conv2D(32, kernel_size=(3,3), activation='relu',padding='same'))
cnn3.add(MaxPool2D(pool_size=(2,2),padding='same'))
cnn3.add(Dropout(0.2))

cnn3.add(Conv2D(64, kernel_size=(3,3), activation='relu',padding='same'))
cnn3.add(MaxPool2D(pool_size=(2,2),padding='same'))
cnn3.add(Dropout(0.2))

cnn3.add(Flatten())
cnn3.add(Dropout(0.2))
cnn3.add(Dense(32,activation='relu'))
cnn3.add(Dropout(0.2))
cnn3.add(Dense(1,activation='sigmoid'))

cnn3.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

image_generatorFullAugment = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.15,
    shear_range=0.3,
    zoom_range=0.30,
    samplewise_center=True,
    horizontal_flip=True,
    samplewise_std_normalization=True
)

image_generatorNoAugment = ImageDataGenerator(
    samplewise_std_normalization=True
)


train = image_generatorFullAugment.flow_from_directory('datasets/train/',
                                                       batch_size=32,
                                                       shuffle=True,
                                                       class_mode='binary',
                                                       target_size=(256,256))

validation = image_generatorFullAugment.flow_from_directory('datasets/validation/',
                                                       batch_size=1,
                                                       shuffle=True,
                                                       class_mode='binary',
                                                       target_size=(256,256))

test = image_generatorNoAugment.flow_from_directory('datasets/test/',
                                                       batch_size=1,
                                                       shuffle=False,
                                                       class_mode='binary',
                                                       target_size=(256,256))

r100 = cnn3.fit(train,
                epochs = 5,
                validation_data=validation,
#                class_weight=class_weight,
                batch_size=64,
                verbose=1)


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(r100.history['loss'],label = 'Loss')
plt.plot(r100.history['val_loss'],label='Val_Loss')
plt.legend()
plt.title('Loss Evolution')


plt.subplot(2,2,2)
plt.plot(r100.history['accuracy'],label='Accuracy')
plt.plot(r100.history['val_accuracy'],label='Val_Accuracy')
plt.legend()
plt.title('Accuracy Evolution')

plt.show()
plt.close()


