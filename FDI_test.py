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
    mask_full_ori = []
    maskrcnn_ori = []
    label_rcnn_ori = []
    label_ori = []
    maskrcnn_full_ori = []
    for _imgname in tqdm.tqdm(datalist):
        temp = []
        _templabel = []
        temask_ori = []
        temask_ori_full = []
        temask_rcnn = []
        temaskrcnn_full = []
        telabel_rcnn = []
        filename = os.path.basename(_imgname)

        # print(filename, end=' ')
        file, _format = os.path.splitext(filename)
        if filename in list_flip:
            img = cv2.imread(imgdir+filename)
            img = cv2.rotate(img, cv2.ROTATE_180)
        else:
            img = cv2.imread(imgdir+filename)
        _label = os.path.join(labeldir+file+'.pkl')
        # print(_label)
        inf = pickle.load(open((_label),'rb') )
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
            cv2.imwrite(f'./check/{os.path.splitext(_imgname)[0]}_{i}_{la}.jpg', resultmask)
            temask_ori.append(resultmask)          ##   只有牙齒
            mask_ori_iou.append(_mask[:, :, i].astype('uint8'))          ##   拿來求IOU的牙齒
            temask_ori_full.append(ori_mask)

        mask_full_ori.append(temask_ori_full)       
        mask_ori.append(temask_ori)
        label_ori.append(_templabel)
        

        # MaskRcnn
        # print(maskdir + file + '.npy')
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
                # print(_max)
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
        match, _ = hungarian(iou)
        # print(len(_iou), len(temp), len(match))
        for midx in match:
            # print(_templabel[midx[1]])
            telabel_rcnn.append(_templabel[midx[1]])

        maskrcnn_ori.append(temask_rcnn)
        maskrcnn_full_ori.append(temaskrcnn_full)
        label_rcnn_ori.append(telabel_rcnn)
                    
            
    #     img = tf.convert_to_tensor(resultmask, dtype=tf.float32)
    #     img = tf.image.resize(img, (image_size, image_size))
    #     img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        imglist.append(img)

    return imglist, mask_ori, mask_full_ori, label_ori, maskrcnn_ori, maskrcnn_full_ori, label_rcnn_ori


##================== Read The File ====================

file = open('FDI_cross_list.pkl', 'rb')
cross_list = pickle.load(file)
part = 0
train_list, val_list = cross_list['train'][part], cross_list['val'][part]

all_list = os.listdir(imgdir)
all_img, all_mask_doctor, all_mask_full_doctor, all_label_doctor, all_mask_maskrcnn, all_mask_maskrcnn_full, all_label_maskrcnn = read_data(all_list, imgdir, labeldir)


#In[]

from sklearn.metrics import f1_score

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


def fit(train, val, model, _weight, callback,  epoch=200):
    history=model.fit(
      train,
      validation_data=val,
      epochs=epoch,
      callbacks=[callback],
      class_weight = _weight
    )
    
    return history

def plot_learning_curve(_history, i, epochs = 100):
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
    
    plt.show()


# def find_threshold(img_list):

#     for idx, img in enumerate(img_list):


def find_center(img):
    tempimg = np.copy(img)
    nonblack = np.argwhere(tempimg != 0)
    x_min, x_max = min(nonblack[:, 1]), max(nonblack[:, 1])
    y_min, y_max = min(nonblack[:, 0]), max(nonblack[:, 0])
    x_center = np.mean([x_min, x_max])
    y_center = np.mean([y_min, y_max])

    return x_center, y_center

def sortmask(mask, maskfull, label):
    center = []
    new_maskrcnn_full = []
    new_maskrcnn = []
    new_label = []
    for idx, ma in enumerate(maskfull):
        temp = []
        for m in ma:
            x_cen, y_cen = find_center(m)
            temp.append(x_cen)
        center.append(temp)
    
    for idx, cen in enumerate(center):
        s = np.argsort(cen)
        new_maskrcnn_full.append(np.array(maskfull[idx])[s])
        new_maskrcnn.append(np.array(mask[idx])[s])
        new_label.append(np.array(label[idx])[s])
    
    return new_maskrcnn_full, new_maskrcnn, new_label



def fdi_sequence(k):
    end = k
    seq = []
    U = np.array([3, 3, 3, 2, 2, 1, 0, 0, 0, 0, 1, 2, 2, 3, 3, 3])
    while end <= 16:
        seq.append(U[end-k:end])
        end += 1        
 
    return seq

def HMM(pr, pr_class):
#     max_pr = np.max(pr, axis=1)
    gt_seq = fdi_sequence(len(pr_class))
    max_score = 0
    # print(pr, pr_class)
    M = np.array([[0.8, 0.15, 0.05, 0, 0],
                  [0.15, 0.7, 0.1, 0.05, 0],
                  [0.05, 0.1, 0.7, 0.15, 0],
                  [0, 0.05, 0.15, 0.8, 0],
                  [0, 0, 0, 0, 0]])
    for seq in gt_seq:
        temp = 0
        for i in range(len(pr_class)):
#             temp += max_pr[i] * M[seq[i]][pr_class[i]]
            if pr_class[i] == seq[i]:
                temp += pr[i] * M[pr_class[i]][pr_class[i]]
            else:
                temp += pr[i] * M[seq[i]][pr_class[i]]
        # print(temp, seq)
        if temp > max_score:
            max_score = temp
            final_seq = seq
#         print("Seq:", seq ,", score = ", temp)
#     print("Final_seq:", final_seq, ", Max_score:", max_score , "\n")
    
    return final_seq

def HMM_adjust(model, test):
    final_pred = []
    i = 0
    for data in test:
        pred = model.predict(np.array(data))
        # pred = model.predict(data)
        max_pr = np.max(pred, axis=1)
        pred_class = np.argmax(pred, axis = 1)
        print('before:', pred_class)
        adjust_pred = HMM(max_pr, pred_class)
        adjust_pred = list(adjust_pred)
        print(i,',', adjust_pred)
        i+= 1
        final_pred += adjust_pred
    
    return final_pred

#=======================Lost Teeth Adjust================================

def lost_teeth(mask, teeth_center, kmeansModel, part=None):
    lostlist = []
    # threshold = 2*std
    for idx, (ma, cen) in enumerate(zip(mask, teeth_center)):
        temp = []
        for m, c in zip(ma, cen):
            # x1_center, y1_center = find_center(ma[i])
            # x_center, y_center = find_center(m)
            x_center, y_center = c
            tempimg = np.copy(m)
            nonblack = np.argwhere(tempimg != 0)
            x_min, x_max = min(nonblack[:, 1]), max(nonblack[:, 1])
            y_min, y_max = min(nonblack[:, 0]), max(nonblack[:, 0])
#             tempimg = np.copy(m)
#             nonblack = np.where(tempimg[y][:, 0] != 0)[0].tolist()
#             x_min, x_max = min(nonblack), max(nonblack)
            temp.append([x_center, (x_max-x_min)])
        # print(temp)
        losttemp = []
        for i in range(len(temp)-1):
            
#             if i == 0:
#                 if (temp[i][0]-avg) > threshold:
#                     losttemp.append([0])
#                 if (temp[i+1][0] - temp[i][0] - avg) > threshold:
#                     losttemp.append([i, i+1])
#             elif i == (len(temp)-1):
#                 if (ma.shape[2] - temp[i][0]) > threshold:
#                     print('last')
#                     losttemp.append([len(temp)-1])
#             else: 
#                 dis = temp[i+1][0] - temp[i][0]
# #                 print(dis)
#                 if dis-avg > threshold:
#                     losttemp.append([i, i+1])
            
            dis = temp[i+1][0] - temp[i][0]
            width = temp[i+1][1] + temp[i][1]
            X = np.array([[dis, width]])
            # print(X.shape)
            y = kmeansModel.predict(X)
            _bool = 0
            if part == 3:
                _bool = 1
            if y == _bool:
                losttemp.append([i, i+1])

        lostlist.append(losttemp)
        print(f'Filename:{all_list[idx]}',  losttemp)

    return lostlist

def lost_adjust(pr, pred_class, lost):
    t = pred_class.tolist()
    s = pr.tolist()
    idx = 0
    for l in lost:
        if len(l) == 1:
            if l[0] == 0:
                t.insert(0, 4)
                s.insert(0, 0)
            else:
                t.append(4)
                s.append(0)
        if len(l) == 2:
            t.insert(l[1]+idx, 4)
            s.insert(l[1]+idx, 0)
        idx += 1
    return np.array(t), np.array(s)


def New_HMM_adjust(model, test, _lostlist):
    final_pred = []
    i = 0
    for idx, data in enumerate(test):
        pred = model.predict(np.array(data))
        max_pr = np.max(pred, axis=1)
        pred_class = np.argmax(pred, axis = 1)
        lost_pos = _lostlist[idx]
        lost_teeth_adjust, pr = lost_adjust(max_pr, pred_class, lost_pos)
        print("Before teeth:")
        print(lost_teeth_adjust)
        adjust_pred = HMM(pr, lost_teeth_adjust)
        new_pred = list(adjust_pred)
        for l in lost_pos:
            if len(l) == 1:
                if l[0] == 0:
                    del new_pred[0]
                else:
                    new_pred.pop()
            if len(l) == 2:
                del new_pred[l[1]]
        print("Final teeth:")
        print(i,',', new_pred)
        i+= 1
        final_pred += new_pred
    
    return final_pred

def confu_matrix(gt, test, filename:str):
    plt.clf()
    conf_numpy = confusion_matrix(gt, test)
    conf_df = pd.DataFrame(conf_numpy, index=[0,1,2,3] ,columns=[0,1,2,3])  #将矩阵转化为 DataFrame
    conf_fig = sn.heatmap(conf_df, annot=True, fmt="d", cmap="BuPu", annot_kws={"fontsize":16})  #绘制 heatmap
    plt.xlabel("Pred", rotation = 0, fontsize = 16)
    plt.yticks(rotation= 0)
    plt.ylabel("GT", rotation = 0, fontsize = 16)
    plt.savefig(f'{filename}.png')
    # plt.show()


def find_teeth_center(img_list):
    center_list = []
    for img in img_list:
        temp = np.zeros((len(img), 2), int)
        for idx, teeth in enumerate(img):
            # print(teeth.shape)
            gimg = cv2.cvtColor(teeth, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(gimg, 1, 255, cv2.THRESH_BINARY)
    # h1, w1 = img.shape[:]
    # binary, contours, hierachy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAI N_APPROX_SIMPLE)
    # X_center = []
    # Y_center = []
    # image = np.zeros([h1, w1])
            M = cv2.moments(binary)
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])
    # print(M)
    # for i in range(len(contours)):
        # M = cv2.moments(contours[i])
        # print(M)
        # center_x = int(M['m10'] / M['m00'])
        # center_y = int(M['m01'] / M['m00'])
        # X_center.append(center_x)
        # Y_center.append(center_y)
        # cv2.drawContours(image, contours, i, 255, 10)
            temp[idx] = [center_x, center_y]
            # cv2.circle(teeth, (center_x, center_y), 7, 128, -1)
            # plt.imshow(teeth)
            # plt.show()
        center_list.append(temp)
        # m_image = cv2.circle(image, (center_x, center_y), 7, 128, -1)
    
    return center_list





def teeth_dis(teeth_center, mask):
    dis_list = []
    width_list = []
    count = 0
    '''
    for mask


    for img in new_mask_full:
        x_cen, y_cen = find_center(img[0])
        dis_list.append(x_cen)
        for idx in range(img.shape[0]-1):
            x1_cen, y1_cen = find_center(img[idx])
            x2_cen, y2_cen = find_center(img[idx+1])
            dis_list.append(abs(x1_cen - x2_cen))
        x_cen, y_cen = find_center(img[-1])
        dis_list.append(img.shape[2]-x_cen)
    '''

    '''
    for teeth
    '''
    for img_center, _mask in zip(teeth_center, mask):
        for i in range(img_center.shape[0]):
            tempimg = np.copy(_mask[i])
            nonblack = np.argwhere(tempimg != 0)
            x1_min, x1_max = min(nonblack[:, 1]), max(nonblack[:, 1])
            y1_min, y1_max = min(nonblack[:, 0]), max(nonblack[:, 0])

            for j in range(i+1, img_center.shape[0]):
                tempimg = np.copy(_mask[j])
                nonblack = np.argwhere(tempimg != 0)
                x2_min, x2_max = min(nonblack[:, 1]), max(nonblack[:, 1])
                y2_min, y2_max = min(nonblack[:, 0]), max(nonblack[:, 0])
                dis = img_center[j][0] - img_center[i][0]
                width = (x2_max - x2_min) + (x1_max - x1_min)
                
                # print(dis_width)
                dis_list.append(dis)
                width_list.append(width) 

    # plt.scatter(dis_list, width_list, c='red', s=1)
    # plt.xlabel('Distance')
    # plt.ylabel('Width')
    # plt.show()

    # avg = np.sum(dis_list) / len(dis_list)
    # std = np.std(dis_list)

    return dis_list, width_list



#In[]
#In[]

# avg_dis = [(i-avg) for i in dis_list]
# for idx, i in enumerate(avg_dis):
#     if i > 2*std:
#         print(idx, i)
#In[]
cross_list['val_idx'][0]
#In[]
# find_threshold(all_mask_full_doctor)
new_mask_full, new_mask_ori, new_label_ori = sortmask(all_mask_maskrcnn, all_mask_maskrcnn_full, all_label_maskrcnn)
# new_mask_full, new_mask_ori, new_label_ori = sortmask(all_mask_doctor, all_mask_full_doctor, all_label_doctor)

'''
Lost Teeth test
'''
all_lost_list = []
from sklearn.cluster import KMeans
#In[]
for part in range(4):
    train_idx = cross_list['train_idx'][part]
    val_idx = cross_list['val_idx'][part]
    x_train_full = list(np.array(new_mask_full)[train_idx])
    teeth_center = find_teeth_center(x_train_full)
    dis_list, width_list = teeth_dis(teeth_center, x_train_full)
# x_val_idx = np.array(new_mask_ori)[val_idx]
# y_val_idx = np.array(new_label_ori)[val_idx]


    X_train = np.array([dis_list, width_list]).T
    kmeansModel = KMeans(n_clusters=2, random_state=46).fit(X_train)



    # X= np.array([dis_list, width_list]).T
    y_pred = kmeansModel.predict(X_train)
    # # y_pred
    # #In[]
    x0 = X_train[y_pred == 0]
    x1 = X_train[y_pred == 1]

    centers = kmeansModel.cluster_centers_
    plt.clf()
    plt.scatter(x0[:, 0], x0[:, 1], c='green', s=1)
    plt.scatter(x1[:, 0], x1[:, 1], c='blue', s=1)
    plt.xlabel('Distance')
    plt.ylabel('Width')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    # plt.show()
    plt.savefig(f'{part}.png')

    x_val_full = list(np.array(new_mask_full)[val_idx])
    teeth_center = find_teeth_center(x_val_full)
    dis_list, width_list = teeth_dis(teeth_center, x_val_full)
    lost_list = lost_teeth(x_val_full, teeth_center, kmeansModel, part)
    all_lost_list.append(lost_list)
#In[]
# all_lost_list



#In[]
# for filename, img, teeth_list, center_list in zip(all_list, all_img, new_mask_full, teeth_center):
#     for mask, center in zip(teeth_list, center_list):
#         tempimg = np.copy(mask)
#         nonblack = np.argwhere(tempimg != 0)
#         x1_min, x1_max = min(nonblack[:, 1]), max(nonblack[:, 1])
#         y1_min, y1_max = min(nonblack[:, 0]), max(nonblack[:, 0])
#         cv2.rectangle(img, (x1_min, y1_min), (x1_max, y1_max), (0, 0, 255), 2)
#         cv2.circle(img, (center[0], center[1]), radius=5, color=(0, 0, 255), thickness=5)

#     cv2.imwrite(f'./weights/{filename}', img)   



#In[]
# len(dis_list)
#In[]
# avg = np.sum(dis_list) / len(dis_list)
# std = np.std(dis_list)
# all_lost_list = lost_teeth(new_mask_full, avg, std)
# all_lost_list = lost_teeth(new_mask_full, teeth_center, kmeansModel)
# print(lost_list)
#In[]
# np.unique(new_mask_full[0][0])
# ret, binary = cv2.threshold(new_mask_full[0][0],1, 255, cv2.THRESH_BINARY)
# plt.imshow(binary)
# plt.show()

# train_idx = cross_list['train_idx'][0]
# val_idx = cross_list['val_idx'][0]
# x_val_idx = np.array(new_mask_ori)[val_idx]
# a = x_val_idx[0].squeeze()
# list(x_val_idx[0])
#In[]
from sklearn.cluster import KMeans
all_model_name = ['LeNet5', 'VGG16', 'MobileNet', 'ResNet50']
_type = 'aug'
class_wt = 'class_wt'
training_data = 'train_doctor_image'
testing_data = 'maskrcnn'


for model_id in range(4):
    model_name = all_model_name[model_id]
    f1_score_rcnn_ori = 0
    f1_score_rcnn_new = 0
    f1_score_rcnn_final = 0
    gt_img = []
    y_gt = []
    y_pred = []
    y_pred_hmm = []
    y_pred_final = []
    for part in range(4):
        x_val = []
        y_val = []
        x_val_hmm = []
        lost_list = []
        model = tf.keras.models.load_model(f'./weights/{training_data}/{class_wt}/{_type}/{model_name}/best_model_{model_name}_{part}.hdf5')
        train_idx = cross_list['train_idx'][part]
        val_idx = cross_list['val_idx'][part]
        x_val_idx = np.array(new_mask_ori)[val_idx]
        # print(x_val_idx.)
        y_val_idx = np.array(new_label_ori)[val_idx]

        ### ==== Lost teeth detect ==== ###
        # x_train_full = list(np.array(new_mask_full)[train_idx])
        # teeth_center = find_teeth_center(x_train_full)
        # dis_list, width_list = teeth_dis(teeth_center, x_train_full)
        # X_train = np.array([dis_list, width_list]).T
        # kmeansModel = KMeans(n_clusters=2, random_state=46).fit(X_train)

        # x_val_full = list(np.array(new_mask_full)[val_idx])
        # teeth_center = find_teeth_center(x_val_full)
        # dis_list, width_list = teeth_dis(teeth_center, x_val_full)
        # lost_list = lost_teeth(x_val_full, teeth_center, kmeansModel)
        ### ====
        lost_list = all_lost_list[part]


        # lost_list_idx = np.array(all_lost_list)[val_idx]
        for i in range(len(x_val_idx)):
            x_val += list(x_val_idx[i])
            gt_img += list(x_val_idx[i])
            x_val_hmm.append(list(x_val_idx[i]))
            # lost_list.append(lost_list_idx[i])
            y_val += list(y_val_idx[i])
            y_gt += list(y_val_idx[i])
        print(lost_list)

        if model_id == 2:
            for i in range(len(x_val)):
                x_val[i] = cv2.resize(x_val[i], (128, 128), interpolation=cv2.INTER_CUBIC)

            for x_v in x_val_hmm:
                for j in range(len(x_v)):
                    x_v[j] = cv2.resize(x_v[j], (128, 128), interpolation=cv2.INTER_CUBIC)

        x_val = np.array(x_val)
        # print(x_val.shape)
        # print(x_val.shape)
        # y_val = np.array(y_val)

        pred_score = model.predict(x_val)
        pred_class = np.argmax(pred_score, axis=1)
        y_pred.extend(pred_class)
        # confu_matrix(y_val, pred_class, f'./weights/{training_data}/{class_wt}/{_type}/{model_name}/{model_name}_{testing_data}_ori_{part}')
        # # print(y_val)

        # score = f1_score(y_val, pred_class, average = 'macro')
        # print('Ori MaskRcnn label score = ', score)
        # f1_score_rcnn_ori += score
        # print(len(x_val[0]))
        pred_class_new = HMM_adjust(model, x_val_hmm)
        y_pred_hmm.extend(pred_class_new)
        # print(pred_class_new)
        # confu_matrix(y_val, pred_class_new, f'./weights/{training_data}/{class_wt}/{_type}/{model_name}/{model_name}_{testing_data}_hmm_{part}')
        # score = f1_score(y_val, pred_class_new , average = 'macro')
        # print('New MaskRcnn label score = ', score, '\n')
        # f1_score_rcnn_new += score

        pred_class_final = New_HMM_adjust(model, x_val_hmm, lost_list)
        y_pred_final.extend(pred_class_final)
        # confu_matrix(y_val, pred_class_final, f'./weights/{training_data}/{class_wt}/{_type}/{model_name}/{model_name}_{testing_data}_final_{part}')
        # score = f1_score(y_val, pred_class_final , average = 'macro')
        # print('Final MaskRcnn label score = ', score, '\n')
        # f1_score_rcnn_final += score

    print(y_gt)
    print(y_pred)
    print(y_pred_hmm)
    print(y_pred_final)


    confu_matrix(y_gt, y_pred, f'./weights/{training_data}/{class_wt}/{_type}/{model_name}/{model_name}_{testing_data}_ori')
    f1_score_rcnn_ori = f1_score(y_gt, y_pred, average = 'macro')
    print('Ori MaskRcnn label score = ', f1_score_rcnn_ori)

    confu_matrix(y_gt, y_pred_hmm, f'./weights/{training_data}/{class_wt}/{_type}/{model_name}/{model_name}_{testing_data}_hmm')
    f1_score_rcnn_new = f1_score(y_gt, y_pred_hmm , average = 'macro')
    print('New MaskRcnn label score = ', f1_score_rcnn_new, '\n')


    confu_matrix(y_gt, y_pred_final, f'./weights/{training_data}/{class_wt}/{_type}/{model_name}/{model_name}_{testing_data}_final')
    f1_score_rcnn_final = f1_score(y_gt, y_pred_final , average = 'macro')
    print('Final MaskRcnn label score = ', f1_score_rcnn_final, '\n')


    print(f'{model_name} Avg Ori MaskRcnn label score = ', f1_score_rcnn_ori)
    print(f'{model_name} Avg New MaskRcnn label score = ', f1_score_rcnn_new)
    print(f'{model_name} Avg Final MaskRcnn Label score= ', f1_score_rcnn_final)


    # Draw False Positive
    plt.figure(figsize = (100, 100))
    count = 1
    # gt_img = [j for i in all_mask_ori for j in i]
    print(len(gt_img))
    for idx, (gt, pred) in enumerate(zip(y_gt, y_pred_final)):
        # print(idx, gt, pred)
        # if(gt != pred):
        ax = plt.subplot(20, 10, count)
        img = gt_img[idx]
        count += 1
        plt.imshow(img)
        plt.title("GT: "+str(gt)+" / pred: "+str(pred))
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f'./weights/{training_data}/{class_wt}/{_type}/{model_name}/{model_name}_{testing_data}_all.png')
# plt.show()
