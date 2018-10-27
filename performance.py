import numpy as np
import pandas as pd
import os
import cv2
from keras import backend as K
from sklearn.metrics import jaccard_similarity_score


# Lesion Classification: Training Labels
training_labels_csv = "datasets/ISIC-2017_Training_Part3_GroundTruth.csv"
validation_labels_csv = "datasets/ISIC-2017_Validation_Part3_GroundTruth.csv"
test_labels_csv = "datasets/ISIC-2017_Test_v2_Part3_GroundTruth.csv"

# Lesion Segmentation: Training Image and Mask
training_folder = "datasets/ISIC-2017_Training_Data"
training_mask_folder = "datasets/ISIC-2017_Training_Part1_GroundTruth"
# Lesion Segmentation: Validation Image
validation_folder = "datasets/ISIC-2017_Validation_Data"
validation_mask_folder = "datasets/ISIC-2017_Validation_Part1_GroundTruth/"
validation_pred_folder = "results/ISIC-2017_Validation_Predicted_backup/model1/"
# Lesion Segmentation: Test Image
test_folder = "datasets/ISIC-2017_Test_v2_Data"
test_mask_folder = "datasets/ISIC-2017_Test_v2_Part1_GroundTruth/"
test_pred_folder = "results/ISIC-2017_Test_v2_Predicted_backup/model1/"


smooth_default = 1.

def dice_coef(y_true, y_pred, smooth = smooth_default, per_batch = True):
    if not per_batch:
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    else: 
        y_true_f = K.batch_flatten(y_true)
        y_pred_f = K.batch_flatten(y_pred)
        intersec = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
        union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
        return K.mean(intersec / union)
    
def jacc_coef(y_true, y_pred, smooth = smooth_default):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)
    
def jacc_loss(y_true, y_pred):
    return -jacc_coef(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
    
def dice_jacc_single(mask_true, mask_pred, smooth = smooth_default):
    bool_true = mask_true.reshape(-1).astype(np.bool)
    bool_pred = mask_pred.reshape(-1).astype(np.bool)
    if bool_true.shape != bool_pred.shape:
        raise ValueError("Masks of different sizes.")

    bool_sum = bool_true.sum() + bool_pred.sum()
    if bool_sum == 0:
        print("Empty mask")
        return 0,0
    intersec = np.logical_and(bool_true, bool_pred).sum()
    dice = 2. * intersec / bool_sum
    jacc = jaccard_similarity_score(bool_true.reshape((1, -1)), bool_pred.reshape((1, -1)), normalize=True, sample_weight=None)
    return dice, jacc

def dice_jacc_mean(mask_true, mask_pred, smooth = smooth_default):
    dice = 0
    jacc = 0
    for i in range(mask_true.shape[0]):
        current_dice, current_jacc = dice_jacc_single(mask_true=mask_true[i],mask_pred=mask_pred[i], smooth= smooth)
        dice = dice + current_dice
        jacc = jacc + current_jacc
    return dice/mask_true.shape[0], jacc/mask_true.shape[0]

def list_from_folder(image_folder):
    image_list = []
    for image_filename in os.listdir(image_folder):
        if image_filename.endswith(".png"):
            image_list.append(image_filename)
    print(("Found {} images.".format(len(image_list))))
    return image_list


print("Calculating Jaccard Similarity Score for Validation Set")
val_mask_list = list_from_folder(validation_mask_folder)
df_val = pd.read_csv(validation_labels_csv)
jacc_val_list = []
dice_val_list = []
for i in range(len(val_mask_list)):
    mask_true = cv2.imread(validation_mask_folder+str(val_mask_list[i]))
    mask_pred = cv2.imread(validation_pred_folder+str(val_mask_list[i]))
    dice, jacc = dice_jacc_single(mask_true, mask_pred)
    jacc_val_list.append(jacc)
    dice_val_list.append(dice)
df_val['jacc'] = jacc_val_list
df_val['dice'] = dice_val_list
print(df_val.head())
df_val.to_csv('val.csv', encoding='utf-8', index=False)


print("Calculating Jaccard Similarity Score for Test Set")
test_mask_list = list_from_folder(test_mask_folder)
df_test = pd.read_csv(test_labels_csv)
jacc_test_list = []
dice_test_list = []
for i in range(len(test_mask_list)):
    mask_true = cv2.imread(test_mask_folder+str(test_mask_list[i]))
    mask_pred = cv2.imread(test_pred_folder+str(test_mask_list[i]))
    dice, jacc = dice_jacc_single(mask_true, mask_pred)
    jacc_test_list.append(jacc)
    dice_test_list.append(dice)
df_test['jacc'] = jacc_test_list
df_test['dice'] = dice_test_list
print(df_test.head())
df_test.to_csv('test.csv', encoding='utf-8', index=False)
