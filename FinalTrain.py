# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

n_train = 75481
n_test = 11189
batch_size = 32

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (256,256,3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(16, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Conv2D(8, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 125, activation = 'softmax'))
# classifier.add(Activation(tf.nn.softmax))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('sketchyTrain/',
                                                 target_size = (256, 256),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('sketchyTest/',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'categorical')

from keras.utils import plot_model
plot_model(classifier, to_file='modelfinal125.png')

from IPython.display import display
from PIL import Image

classifier.fit_generator(training_set,
                         steps_per_epoch = 2*(n_train/batch_size),
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000)

from keras.models import load_model

classifier.save('sketchmodel125final.h5')  # creates a HDF5 file 'my_model.h5'
classifier.save('sketchmodel125final.hdf5')