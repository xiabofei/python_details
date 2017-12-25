import os
import sys
import numpy as np
import pandas as pd

from keras.applications import vgg16, vgg19, resnet50, inception_v3, xception
from keras.layers import Activation, Dropout, Dense, Input, Conv2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD

from my_image import ImageDataGenerator
from utilizers import multi_log_loss

###################################################
## Settings
###################################################

# define the structure of the top model 
def get_top_model(x, dpr):
    
    x = GlobalMaxPooling2D()(x)
    x = Dropout(dpr)(x)
    x = Dense(17, activation = 'sigmoid')(x)
    
    return x

# parameters
base_model_type = 'xception' #vgg16, vgg19, resnet50, inceptionv3, xception
#base_model_type = sys.argv[1]
img_size = 299
batch_size = 32
#n1 = 512
#n2 = 256
dpr = np.random.rand() * 0.4

nb_epoch_pretrain = 200
nb_epoch_tune = 200
NBAGS = 8

re_train = True
if(len(sys.argv) > 2):
    if(sys.argv[2] == 'False'):
        re_train = False

print('Using base model: ' + base_model_type)
print('Re-train the model: %s'%re_train)

stamp = base_model_type + '_%d_basic_max_%.2f'%(img_size, dpr)
print(stamp)

# dimensions of our images
img_height, img_width = img_size, img_size

# path to the data.
input_dir = '../input/'
train_data_dir = 'input/train/'
val_data_dir = 'input/validation/'
test_data_dir = '../input/test/'
model_dir = 'model/'

###################################################
## Model Structure
###################################################

# creat the base pre-trained model
input_tensor = Input(shape = (img_height, img_width, 3))

if base_model_type == 'vgg16':
    base_model = vgg16.VGG16(input_tensor = input_tensor, 
            weights = 'imagenet', include_top = False)
elif base_model_type == 'vgg19':
    base_model = vgg19.VGG19(input_tensor = input_tensor, 
            weights = 'imagenet', include_top = False)
elif base_model_type == 'resnet50':
    base_model = resnet50.ResNet50(input_tensor = input_tensor, 
            weights = 'imagenet', include_top = False)
elif base_model_type == 'inceptionv3':
    base_model = inception_v3.InceptionV3(input_tensor = input_tensor, 
            weights = 'imagenet', include_top = False)
elif base_model_type == 'xception':
    base_model = xception.Xception(input_tensor = input_tensor, 
            weights = 'imagenet', include_top = False)

# creat the top model
top_model = get_top_model(base_model.output, dpr)

# creat the model we will train
model = Model(inputs = base_model.input, outputs = top_model)

###################################################
## Prepare lables
###################################################

labels = pd.read_csv(input_dir+'train_labels.csv')
label_dict = labels.set_index('image_name')['ids'].apply(lambda x: [int(i) for i in x.split()]).to_dict()
id2tag = pd.read_csv(input_dir+'id_tag.csv')

###################################################
## Training and Validation Data
###################################################

# prepare the data generator
def preprocess_input1(x):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    return x
    
def preprocess_input2(x):
    x /= 255.
    x -= 0.5
    x *= 2.0
    return x

if(base_model_type in ['vgg16', 'vgg19', 'resnet50']):
    preprocessor = preprocess_input1
else:
    preprocessor = preprocess_input2

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        rotation_range = 5,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        shear_range = 0.1,
        zoom_range = 0.1,
        channel_shift_range = 10,
        horizontal_flip = True,
        vertical_flip = True,
        preprocessing_function = preprocessor)
train_generator = train_datagen.flow_from_directory(train_data_dir,
        label_dict = label_dict,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True)

val_datagen = ImageDataGenerator(preprocessing_function = preprocessor)
val_generator = val_datagen.flow_from_directory(val_data_dir,
        label_dict = label_dict,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = False)

if(re_train):
    ###################################################
    ## Pretraining the Top Model
    ###################################################

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
            metrics = ['accuracy'])
    
    #model.summary()
    
    # use a early stopping strategy
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)
    model_file = model_dir + stamp + '_pretrain.h5'
    model_checkpoint = ModelCheckpoint(model_file, save_best_only = True, save_weights_only = True)

    # pretrain the model
    model.fit_generator(
            train_generator,
            steps_per_epoch = train_generator.n/batch_size,
            epochs = nb_epoch_pretrain,
            validation_data = val_generator,
            validation_steps = val_generator.n/batch_size,
        	callbacks = [early_stopping, model_checkpoint],
            max_q_size = 384, workers = 3, verbose = 2)

    # finishing pre-training
    model.load_weights(model_file)
    os.remove(model_file)
    print('Pre-training finished')

    ###################################################
    ## Fine-tuning the Model with Adam
    ###################################################

    for layer in model.layers:
        layer.trainable = False

    if base_model_type == 'vgg16':
        for layer in model.layers[12:]:  # fine-tune 2 layers
            layer.trainable = True
    elif base_model_type == 'vgg19':
        for layer in model.layers[18:]:
            layer.trainable = True
    elif base_model_type == 'resnet50':
        for layer in model.layers[142:]:  # fine-tune 4 blocks
            layer.trainable = True
    elif base_model_type == 'inceptionv3':
        for layer in model.layers[194:]:
            layer.trainable = True
    elif base_model_type == 'xception':
        for layer in model.layers[106:]:  # fine-tune 3 blocks
            layer.trainable = True

    model.compile(optimizer = 'adam',
            loss = 'binary_crossentropy', metrics = ['accuracy'])

    #model.summary()

    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)
    model_file = model_dir + stamp + '_tuned.h5'
    model_checkpoint = ModelCheckpoint(model_file, save_best_only = True, save_weights_only = True)

    hist = model.fit_generator(
            train_generator,
            steps_per_epoch = train_generator.n/batch_size,
            epochs = nb_epoch_tune,
            validation_data = val_generator,
            validation_steps = val_generator.n/batch_size,
        	callbacks = [early_stopping, model_checkpoint],
            max_q_size = 384, workers = 3, verbose = 2)

    model.load_weights(model_file)
    #os.remove(model_file)
    bst_score_val = min(hist.history['val_loss'])
    print('Adam fine-tuning finished')

    ###################################################
    ## Fine-tuning the Model with SGD
    ###################################################

    for layer in model.layers:
        layer.trainable = False

    if base_model_type == 'vgg16':
        for layer in model.layers[12:]:  # fine-tune 2 layers
            layer.trainable = True
    elif base_model_type == 'vgg19':
        for layer in model.layers[18:]:
            layer.trainable = True
    elif base_model_type == 'resnet50':
        for layer in model.layers[112:]:  # fine-tune 4 blocks
            layer.trainable = True
    elif base_model_type == 'inceptionv3':
        for layer in model.layers[194:]:
            layer.trainable = True
    elif base_model_type == 'xception':
        for layer in model.layers[106:]:  # fine-tune 3 blocks
            layer.trainable = True

    model.compile(optimizer = SGD(lr=0.0001, momentum=0.9),
            loss = 'binary_crossentropy', metrics = ['accuracy'])

    #model.summary()

    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 20)
    model_file = model_dir + stamp + '_tuned.h5'
    model_checkpoint = ModelCheckpoint(model_file, save_best_only = True, save_weights_only = True)

    hist = model.fit_generator(
            train_generator,
            steps_per_epoch = train_generator.n/batch_size,
            epochs = nb_epoch_tune,
            validation_data = val_generator,
            validation_steps = val_generator.n/batch_size,
        	callbacks = [early_stopping, model_checkpoint],
            max_q_size = 384, workers = 3, verbose = 2)

    model.load_weights(model_file)
    #os.remove(model_file)
    bst_score_val = min(hist.history['val_loss'])
    print('SGD fine-tuning finished')

else:
    model.compile(optimizer = SGD(lr=0.0001, momentum=0.9),
            loss = 'binary_crossentropy', metrics = ['accuracy'])
    model_file = model_dir + stamp + '_tuned.h5'
    model.load_weights(model_file)
    print('Pre-trained model loaded')
    hist = model.evaluate_generator(
            val_generator,
            steps = val_generator.n/batch_size,
            max_q_size = 384, workers = 5)
    bst_score_val = hist[0]

###################################################
## Make the Submission
###################################################

for i in range(NBAGS):
    print('Bag %d begin'%i)
    # make submission for validation set
    val_datagen = ImageDataGenerator(
            rotation_range = 3,
            width_shift_range = 0.05,
            height_shift_range = 0.05,
            shear_range = 0.05,
            zoom_range = 0.05,
            channel_shift_range = 5,
            horizontal_flip = True,
            vertical_flip = True,      
            preprocessing_function = preprocessor)
    val_generator = val_datagen.flow_from_directory(val_data_dir,
            label_dict = label_dict,
            target_size = (img_width, img_height),
            batch_size = batch_size*2,
            class_mode = None,
            shuffle = False)
    if(i == 0):
        y_predict_val = model.predict_generator(val_generator,
                steps = np.ceil(val_generator.n*0.5/batch_size)) / NBAGS
    else:
        y_predict_val += model.predict_generator(val_generator,
                steps = np.ceil(val_generator.n*0.5/batch_size)) / NBAGS


    # make submission for test set
    test_datagen = ImageDataGenerator(
            rotation_range = 3,
            width_shift_range = 0.05,
            height_shift_range = 0.05,
            shear_range = 0.05,
            zoom_range = 0.05,
            channel_shift_range = 5,
            horizontal_flip = True,
            vertical_flip = True,
            preprocessing_function = preprocessor)
    test_generator = test_datagen.flow_from_directory(test_data_dir,
            target_size = (img_width, img_height),
            batch_size = batch_size*2,
            class_mode = None,
            shuffle = False)
    if(i == 0):
        y_predict = model.predict_generator(test_generator,
                steps = np.ceil(test_generator.n*0.5/batch_size)) / NBAGS
    else:
        y_predict += model.predict_generator(test_generator,
                steps = np.ceil(test_generator.n*0.5/batch_size)) / NBAGS

# get image names
val_name = [i.split('/')[1].replace('.jpg','') for i in val_generator.filenames]
test_name = [i.split('/')[1].replace('.jpg','') for i in test_generator.filenames]

# get val_y and val_loss
val_y = np.zeros((val_generator.n, 17))
for i in range(val_generator.n):
    val_y[i, label_dict[val_name[i]]] = 1.0
final_score_val = multi_log_loss(val_y, y_predict_val)

print('Val loss before averaging: %.5f'%bst_score_val)
print('Val loss after averaging:  %.5f'%final_score_val)

sub = pd.DataFrame(y_predict_val)
sub.columns = id2tag['tag'].values
sub['image_name'] = val_name
sub.to_csv('val/%s_299/%.5f_'%(base_model_type, final_score_val)+stamp+'.csv', index=False)

sub = pd.DataFrame(y_predict)
sub.columns = id2tag['tag'].values
sub['image_name'] = test_name
sub.to_csv('test/%s_299/%.5f_'%(base_model_type, final_score_val)+stamp+'.csv', index=False)

# rename model file
new_model_file = model_dir + '%s_299/%.5f_'%(base_model_type, final_score_val) + stamp + '_tuned.h5'
os.system('mv %s %s'%(model_file, new_model_file))
