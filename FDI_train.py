#In[]
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import seaborn as sn
import random
from PIL import Image
import openpyxl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, optimizers, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, Convolution2D, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight 
from sklearn.metrics import plot_confusion_matrix
from scipy.optimize import linear_sum_assignment
import copy
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tqdm


imgdir = os.path.join('./doctor_label_image/')
labeldir = os.path.join('./PA_doctor_label1/')
maskdir = os.path.join('./masks/')
t = pickle.load(open((labeldir+ './teeth34.pkl'),'rb') )
image_size = 64


list_flip=[]
wb = openpyxl.load_workbook('./轉180.xlsx')
for name_sheet in wb.sheetnames:
    sheet = wb[name_sheet]
    for column in sheet.columns:
        for cell in column:
            list_flip.extend([cell.value])
list_flip = list_flip[:7]



def bbox(img):
    nonblack = np.argwhere(img != 0)
    x_min, x_max = min(nonblack[:, 1]), max(nonblack[:, 1])
    y_min, y_max = min(nonblack[:, 0]), max(nonblack[:, 0])
    
    return x_min, x_max, y_min, y_max


def resize(img, size):
    h_, w_ = img.shape[:2]
    # print(h_, w_)
    if (size / h_) < (size / w_) :
                
        new_width = int(w_ * float((size / h_)))
        # print(new_width)
        img = cv2.resize(img, (new_width, size), cv2.INTER_CUBIC)
                
    else:
        new_height = int(h_ * float((size / w_)))
        # print(new_height)
        img = cv2.resize(img, (size, new_height), cv2.INTER_CUBIC)

    h_, w_ = img.shape[:2]
    top = int((size - h_) / 2)
    down = int((size - h_ + 1) / 2)
    left = int((size - w_) / 2)
    right = int((size - w_ + 1) / 2)

    img = cv2.copyMakeBorder(img, top, down, left, right, cv2.BORDER_CONSTANT, None, [0, 0, 0])
    return img


def IoU(img1, img2):
    # img2 = cv2.resize(img2, (512, 512), interpolation = cv2.INTER_NEAREST)
    # cv2.imshow('', np.hstack([img1, img2]))
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # exit()
    # x1_min, x1_max, y1_min, y1_max = bbox(img1)
    # x2_min, x2_max, y2_min, y2_max = bbox(img2)
    
    # area1 = (x1_max - x1_min) * (y1_max - y1_min)
    # area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    # ixmin = max(x1_min, x2_min)
    # iymin = max(y1_min, y2_min)
    # ixmax = min(x1_max, x2_max)
    # iymax = min(y1_max, y2_max)
    
    # iw = max(ixmax - ixmin, 0)
    # ih = max(iymax - iymin, 0)
    temp = img1 + img2
    num = np.bincount(temp.flatten())
    # print(len(num))
    union = sum(num[1:])
    if (len(num) > 2):  
        inter = num[2]
    else:
        inter = 0
    
    # inter = iw * ih
    # union = area1 + area2 - inter
    
    _iou = inter / union
    
    return _iou


def hungarian(iou):
        """Hungarian algorithm.
        The difference between Hungarian and greedy is that Hungarian finds a set with min total cost.
        """
        match = []
        unmatch = {
            'tracklets': set(range(iou.shape[0])),
            'detections': set(range(iou.shape[1]))
        }
        unmatch_tracklets = set(range(iou.shape[0]))
        unmatch_dets = set(range(iou.shape[1]))
        row_ind, col_ind = linear_sum_assignment(iou, maximize=True)

        for r, c in zip(row_ind, col_ind):
            # match.append((r, c, dist[r, c]))
            match.append((r, c))
            unmatch['tracklets'].remove(r)
            unmatch['detections'].remove(c)

        return match, unmatch


def cross_validation(data_dir, part=4):
    train_list = []
    train_idx = []
    val_list = []
    val_idx = []
    alldatalist = np.array([i for i in os.listdir(data_dir)])
    # alldatalist = np.array([i for i in glob.glob(os.path.join(imgdir+'*.jpg'))])

    num_data = alldatalist.shape[0]
    
    shuffle_list = np.arange(0, num_data, 1)
    np.random.shuffle(shuffle_list)
    # print(np.random.shuffle(idx_list))
    # shuffle_list = np.random.shuffle(idx_list)
    num_val = num_data // part
    count = 0
    for i in range(0, part-1):
        train = set(copy.deepcopy(shuffle_list))
        val = set(copy.deepcopy(shuffle_list[count:count + num_val]))
        train = train - val
        print(train, val)
        print(list(train), list(val))

        train_list.append(alldatalist[list(train)])
        val_list.append(alldatalist[list(val)])
        train_idx.append(list(train))
        val_idx.append(list(val))

        count += num_val
  
    train_list.append(alldatalist[shuffle_list[0:count]])
    train_idx.append(shuffle_list[0:count])
    val_list.append(alldatalist[shuffle_list[count:]])
    val_idx.append(shuffle_list[count:])

    return train_list, val_list, train_idx, val_idx

def read_data(datalist, imgpath, labelpath):
    imglist = []
    mask_ori = []
    maskrcnn_ori = []
    label_rcnn_ori = []
    label_ori = []
    maskrcnn_full_ori = []
    for _imgname in datalist:
        temp = []
        _templabel = []
        temask_ori = []
        temask_rcnn = []
        temaskrcnn_full = []
        telabel_rcnn = []
        filename = os.path.basename(_imgname)

        print(filename)
        file, _format = os.path.splitext(filename)
        if filename in list_flip:
            img = cv2.imread(imgdir+filename)
            img = cv2.rotate(img, cv2.ROTATE_180)
        else:
            img = cv2.imread(imgdir+filename)
        _label = os.path.join(labeldir+file+'.pkl')
        print(_label)
        inf = pickle.load(open((_label),'rb') )
        font_img = copy.deepcopy(img)       # 印出示意圖用
        # FDI = inf['teethgMask_position']
        
        
        # for fdi in FDI:
        #     dig = fdi % 10
        #     la = None
        #     if dig <= 2:
        #         la = 0
        #     elif dig == 3:
        #         la = 1
        #     elif 4 <= dig <= 5:
        #         la = 2
        #     else:
        #         la = 3
                
        #     label_ori.append(la)
        #     _templabel.append(la)
            
        _mask = inf['label_txt_teethMask']
        FDI = inf['teethgMask_position']

        # 醫生標記
        mask_ori_iou = []
        for i in range(_mask.shape[2]):
            fdi = FDI[i]
            dig = fdi % 10
            if dig <= 2:
                la = 0
            elif dig == 3:
                la = 1
            elif 4 <= dig <= 5:
                la = 2
            else:
                la = 3
            #print(i)
            # maskidx = np.argwhere(_mask[:,:,i] == 1)        
            # resultmask = tomask(img, maskidx)
            x_min, x_max, y_min, y_max = bbox(_mask[:, :, i])
            # print(np.unique(_mask[:, :, i]*255))
            ori_mask = copy.deepcopy(img)
            ori_mask[_mask[:, :, i] == 0] = [0, 0, 0] 
            # resultmask = np.zeros((y_max-y_min, x_max-x_min), dtype=int)

            ### ==== 印出示意圖 ====
            # font_img[_mask[:, :, i] == 1] = [80, 127, 255]
            # cht_FDI = ['Incisor','Canine','Premolar','Molar']
            # cv2.rectangle(font_img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            # text = f"FDI:{cht_FDI[la]}({la})"
            # cv2.putText(font_img, text, (x_min+10, y_min+100), cv2.FONT_HERSHEY_DUPLEX, 1.25, (255, 0, 0), 3 ,cv2.LINE_AA)
            
            # cv2.imshow('', font_img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # exit()


            if abs(x_min-x_max) < 10 or abs(y_min-y_max) < 10:
                resultmask = ori_mask[y_min:y_max, x_min:x_max, :]
                cv2.imwrite(f'./check/defective/{os.path.splitext(_imgname)[0]}_{i}_{la}.jpg', resultmask)
                continue

            
            _templabel.append(la)
            resultmask = ori_mask[y_min:y_max, x_min:x_max, :]
            # resultmask = temp_img[_mask[:, :, i] == 1]
            
            # cv2.imshow('', temp*255)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            # exit()
            resultmask = resize(resultmask, image_size)
            # resultmask = cv2.resize(resultmask, (image_size, image_size), interpolation = cv2.INTER_CUBIC)
            # cv2.imshow('', resultmask)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            # exit()
    #         resultmask = tf.convert_to_tensor(resultmask, dtype=tf.float32)
    #         resultmask = tf.image.resize(resultmask,(image_size, image_size))
            cv2.imwrite(f'./check/{os.path.splitext(_imgname)[0]}_{i}_{la}.jpg', ori_mask)
            temask_ori.append(resultmask)          ##   只有牙齒
            mask_ori_iou.append(_mask[:, :, i].astype('uint8'))          ##   拿來求IOU的牙齒
        
        mask_ori.append(temask_ori)
        label_ori.append(_templabel)
        
        # MaskRcnn
        # print(maskdir + file + '.npy')
        # temp_maskrcnn = np.load(maskdir+file+'.npy')
        file = open(maskdir+file+'.pkl', 'rb')
        temp_maskrcnn = pickle.load(file)['masks']
        _iou = []
        for i in range(temp_maskrcnn.shape[0]):
            temp_label = None
            arr = []
            _max = 0
            # maskidx = np.argwhere(temp_maskrcnn[:, :, i] == 1)
            # teimg = cv2.resize(img, (512, 512), interpolation = cv2.INTER_CUBIC)
            x_min, x_max, y_min, y_max = bbox(temp_maskrcnn[i, :, :])
            ori_maskrcnn = copy.deepcopy(img)
            ori_maskrcnn[temp_maskrcnn[i, :, :] == 0] = [0, 0, 0]
            # print(ori_maskrcnn.shape)
            resultmask = ori_maskrcnn[y_min:y_max, x_min:x_max, :]
            # resultmask = tomask(teimg, maskidx)
            # resultmask = teimg[temp_maskrcnn[:, :, i] == 1]
            resultmask = resize(resultmask, image_size) ##   只有牙齒
            
            # resultmask = cv2.resize(resultmask, (image_size, image_size), interpolation = cv2.INTER_CUBIC)
            for idx, te in enumerate(mask_ori_iou):
                a = IoU(temp_maskrcnn[i, :, :].astype(int), te)
                arr.append(a)
                if a > _max:
                    _max = a
                    
            if _max > 0.5:
                print(_max)
                _iou.append(arr)
                temask_rcnn.append(resultmask)
                temaskrcnn_full.append(ori_maskrcnn)
                
    #             print(a, _iou)
    #             if a > _iou:
    #                 _iou = a
    #                 temp_label = _templabel[idx]
    #         if _iou < 0.5:
    #             continue
    #         print(temp_label)
        iou = np.array(_iou)
        iou = np.resize(iou, (len(_iou), len(mask_ori_iou)))
        match, unmatch = hungarian(iou)
        print(len(_iou), len(temp), len(match), unmatch)
        for midx in match:
            print(_templabel[midx[1]])
            telabel_rcnn.append(_templabel[midx[1]])

        maskrcnn_ori.append(temask_rcnn)
        maskrcnn_full_ori.append(temaskrcnn_full)
        label_rcnn_ori.append(telabel_rcnn)
                    
            
    #     img = tf.convert_to_tensor(resultmask, dtype=tf.float32)
    #     img = tf.image.resize(img, (image_size, image_size))
    #     img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        imglist.append(img)


        # cv2.imwrite(f'./check/FDI/{filename}', font_img)


    return imglist, mask_ori, label_ori, maskrcnn_ori, maskrcnn_full_ori, label_rcnn_ori

#In[]
count = 0
a = glob.glob('./masks/*.pkl')
for fi in a:
    file = open(fi, 'rb')
    temp = pickle.load(file)
    # print(temp['masks'].shape)
    count += (temp['masks'].shape[0])
print(count)

#In[]


file = open('FDI_cross_list.pkl', 'rb')
cross_list = pickle.load(file)
part = 0
train_list, val_list = cross_list['train'][part], cross_list['val'][part]

all_list = os.listdir(imgdir)
# all_list = list(train_list) + list(val_list)
# train_img, train_mask_ori, train_label_ori, train_mask_maskrcnn, train_mask_maskrcnn_full, train_label_maskrcnn = read_data(train_list, imgdir, labeldir)
# val_img, val_mask_ori, val_label_ori, val_mask_maskrcnn, val_mask_maskrcnn_full, val_label_maskrcnn = read_data(val_list, imgdir, labeldir)
all_img, all_mask_ori, all_label_ori, all_mask_maskrcnn, all_mask_maskrcnn_full, all_label_maskrcnn = read_data(all_list, imgdir, labeldir)

#In[]


# np.bincount(all_label_ori)
# os.listdir(imgdir)
# a =[]
# for filename, mask_list, label_ori in zip(all_list, all_mask_ori, all_label_ori):
#     for mask, label in zip(mask_list, label_ori):
#         cv2.imwrite(f'./Data{filename}')
#In[]
# count = 0
# for idx, (a, b) in enumerate(zip(all_mask_maskrcnn, all_label_maskrcnn)):
#     print(idx, b)
#     for c, fdi in zip(a, b):
#         cv2.imwrite(f'./check/{all_list[idx]}_{count}_{fdi}.jpg', c)
#         count += 1
# count



#In[]

def class_weight(_label):
    weight = {}
    _class = np.unique([0,1,2,3]) #标签类别
    print(np.bincount(np.array(_label)))
    w = compute_class_weight(class_weight='balanced', classes=_class, y = _label)
    weight[0] = w[0]
    weight[1] = w[1]
    weight[2] = w[2]
    weight[3] = w[3]
    print(weight)
    return weight

#In[]

def fit(train, val, model, _weight, callback,  epoch=200):
    history=model.fit(
      train,
      validation_data=val,
      epochs=epoch,
      callbacks=[callback],
      class_weight = _weight
    )
    
    return history

def plot_learning_curve(_history, i, epochs = 100, filename=None):
    plt.clf()
    plt.figure(figsize=(32, 16))
    
    for idx in range(i):
        acc = _history[idx].history['accuracy']
        val_acc = _history[idx].history['val_accuracy']

        loss = _history[idx].history['loss']
        val_loss = _history[idx].history['val_loss']

        epochs_range = range(epochs)


        plt.subplot(1, 4, 1)
        plt.plot(epochs_range, acc, label='Epoch ' + str(idx))
        plt.legend(loc='lower right', prop={'size': 16})
        plt.title('Training Accuracy', fontsize=20)
        plt.subplot(1, 4, 2)
        plt.plot(epochs_range, val_acc, label='Epoch ' + str(idx))
        plt.legend(loc='lower right', prop={'size': 16})
        plt.title('Validation Accuracy', fontsize=20)

        plt.subplot(1, 4, 3)
        plt.plot(epochs_range, loss, label='Epoch ' + str(idx))
        plt.legend(loc='upper right', prop={'size': 16})
        plt.title('Training Loss', fontsize=20)
        plt.subplot(1, 4, 4)
        plt.plot(epochs_range, val_loss, label='Epoch ' + str(idx))
        plt.legend(loc='upper right', prop={'size': 16})
        plt.title('Validation Loss', fontsize=20)
    
    plt.savefig(filename)
    # plt.show()


#------------------- Model Type----------------------------

def lenet():
    print("LeNet/n")
    model = tf.keras.Sequential([
        layers.Conv2D(6, (5, 5), activation = 'tanh', input_shape = (64, 64, 3)),
        layers.Dropout(0.5),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (5, 5), activation = 'tanh'),
        layers.Dropout(0.5),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation = 'tanh'),
        layers.Dense(84, activation = 'tanh'),
        layers.Dense(4, activation = 'softmax')
    ])
    
    model.summary()

    return model


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.layers import Input
def _VGG16():
    # print("VGG16/n")
#     model = models.Sequential()
#     weight_decay = 5e-4
    pretrained_model = VGG16(input_shape=(64,64,3), weights = 'imagenet', include_top = False)
#     pretrained_model.trainable = False

#     pretrained_model.trainable = True
#     model.add(Input(shape=(64, 64, 3))) 
#     model.add(Convolution2D(3, (3, 3), padding='same', input_shape=(64, 64, 3)))
#     model.add(LeakyReLU(alpha=0.1))
#     for layer in pretrained_model.layers:
#         model.add(layer)


    model = Flatten(name='flatten')(pretrained_model.output)
#     model = GlobalAveragePooling2D()(model)
    model = Dense(4096, activation='relu', name='fc1')(model)
    model = Dense(4096, activation='relu', name='fc2')(model)
    model = Dense(4, activation='softmax')(model)
    model_vgg_mnist = models.Model(inputs=pretrained_model.input, outputs=model, name='vgg16')
    for layer in pretrained_model.layers:
        layer.trainable = False
        
    model_vgg_mnist.summary()
#     return model
    return model_vgg_mnist



def mobilenet():
    # print("MobileNet/n")
    model = models.Sequential()
    pretrained_model = applications.MobileNet(input_shape=(
    128, 128, 3), include_top = False, weights = 'imagenet')
    pretrained_model.trainable = True
    
#     model.add(Convolution2D(3, (1, 1), padding='same', input_shape=(128, 128, 3)))
#     model.add(LeakyReLU(alpha=0.1))
    for layer in pretrained_model.layers:
        model.add(layer)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.1))
    model.add(Dense(4))
    model.summary()
    return model

from tensorflow.keras.applications.resnet50 import ResNet50
def _resnet50():
    # print("ResNet50/n")
    restnet = ResNet50(input_shape=(64,64,3), weights='imagenet', include_top = False)
    output = restnet.layers[-1].output
    output = keras.layers.Flatten()(output)
    restnet = models.Model(restnet.input, outputs=output)

    model = models.Sequential()
    model.add(restnet)
    model.add(Dense(512, activation='relu', input_dim=(64, 64, 3)))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(4, activation='sigmoid'))
    
    for layer in restnet.layers:
        layer.trainable = False

    model.summary()
    return model




f1_score_mobile_rcnn_ori = 0
f1_score_mobile_rcnn_new = 0
training_data = 'train_doctor_image'
_class_wt = 'no_class_wt'
_type = 'aug'
all_model_name = ['LeNet5', 'VGG16', 'MobileNet', 'ResNet50']
for model_id in range(4):
    # model_id = 0
    model_name = all_model_name[model_id]
    history = []

    for part in range(4):
        x_train = []
        y_train = []
        x_val = []
        x_val_hmm = []
        y_val = []
        train_idx = cross_list['train_idx'][part]
        val_idx = cross_list['val_idx'][part]
        x_train_idx = np.array(all_mask_ori)[train_idx]
        y_train_idx = np.array(all_label_ori)[train_idx]
        # x_train = np.zeros((len(train_idx), image_size, image_size), dtype=int)
        # y_train = np.zeros((len))


        for i in range(len(x_train_idx)):
            
            x_train += x_train_idx[i]
            y_train += y_train_idx[i]
        
        if model_id == 2:
            for i in range(len(x_train)):
                x_train[i] = cv2.resize(x_train[i], (128, 128), interpolation=cv2.INTER_CUBIC)



        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_val_idx = np.array(all_mask_ori)[val_idx]
        y_val_idx = np.array(all_label_ori)[val_idx]
        
        for i in range(len(x_val_idx)):
            x_val += x_val_idx[i]
            x_val_hmm.append(x_val_idx[i])
            y_val += y_val_idx[i]
        
        if model_id == 2:
            for i in range(len(x_val)):
                x_val[i] = cv2.resize(x_val[i], (128, 128), interpolation=cv2.INTER_CUBIC)
        
        x_val = np.array(x_val)
        y_val = np.array(y_val)

        aug = ImageDataGenerator(
            rotation_range=15,
            horizontal_flip=True,
        )
        aug.fit(x_train)


        checkpoint = ModelCheckpoint(f"./weights/{training_data}/{_class_wt}/{_type}/{model_name}/best_model_{model_name}_{part}.hdf5", monitor='val_accuracy', verbose=1,
                save_best_only=True, mode='max', period=1)
            
        stop = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min',
                                    baseline=None, restore_best_weights=True)
        
        # all = [lenet(), _VGG16(), mobilenet(), _resnet50()]
        if model_id == 0:   
            model = lenet()
        if model_id == 1:   
            model = _VGG16()
        if model_id == 2:   
            model = mobilenet()
        if model_id == 3:   
            model = _resnet50()
        # model = lenet()
        # model = _VGG16()
        # model = mobilenet()
        # model = _resnet50()
        # print('Now:', all[model_id])
        # model = all[model_id]


        sgd = optimizers.SGD(learning_rate=0.01)

        model.compile(
            optimizer=sgd,
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        class_wt = class_weight(y_train)
        
        history.append(model.fit(
                                aug.flow(x_train, y_train, batch_size=16), 
                                callbacks=[checkpoint], 
                                epochs=100, 
                                validation_data=(x_val, y_val), 
                                # class_weight=class_wt,
                                )
        )
        # history.append(model.fit(
        #                         x_train,
        #                         y_train, 
        #                         callbacks=[checkpoint], 
        #                         epochs=100, 
        #                         validation_data=(x_val, y_val), 
        #                         class_weight=class_wt,
        #                         batch_size=16,
        #                         )
        #             )
    

    plot_learning_curve(history, 4, filename=f'./weights/{training_data}/{_class_wt}/{_type}/{model_name}/{model_name}.png')