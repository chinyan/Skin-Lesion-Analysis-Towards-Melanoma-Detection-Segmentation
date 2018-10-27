import numpy as np
from os import listdir
import tensorflow as tf
from sklearn.metrics import jaccard_similarity_score
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from metrics import dice_loss, jacc_loss, jacc_coef, dice_jacc_mean,dice_jacc_single

def extract_predicted_mask(predicted_filepath,actual_filepath):
    predicted_img_list = []
    actual_img_mask = listdir(actual_filepath)
    print(actual_img_mask)
    predicted_img_mask_list = listdir(predicted_filepath)
    counter = 0
    total_jac = 0
    total_dice=0
    output_file = open('results.csv','w')
    output_file.write('file_name,Jaccard_Score,Dice_Score\n')
    for img_file in predicted_img_mask_list:
        img = load_img(predicted_filepath+'\\'+img_file)
        predicted_img_mask = img_to_array(img)
        #predicted_img_mask2 = predicted_img_mask.reshape(-1)
        predicted_img_mask2 = predicted_img_mask.flatten()

        img2 = load_img(actual_filepath+img_file)
        actual_img_mask = img_to_array(img2)
        #actual_img_mask2 = actual_img_mask.reshape(-1)

        actual_img_mask2 = actual_img_mask.flatten()

        #jac_score = jaccard_similarity_score(actual_img_mask2, predicted_img_mask2)
        dice_score,jac_score = dice_jacc_single(actual_img_mask,predicted_img_mask)
        output_file.write(img_file+','+str(jac_score)+','+str(dice_score)+'\n')
        #print('j:'+str(jac_score))
        #print('d:'+str(dice_score))
        total_jac = total_jac + jac_score
        total_dice = total_dice + dice_score
        counter=counter+1
    output_file.close()
    average_jac = total_jac/len(predicted_img_mask_list)
    average_dice = total_dice/len(predicted_img_mask_list)
    print('jac avg:'+str(average_jac))
    print('dice avg:' + str(average_dice))
    return average_jac


root_path_validation_predicted = 'C:\\NUS_ISS\\KE5108_DEVELOPING_INTELLIGENT_SYSTEMS_FOR_BUSINESS ANALYTICS\\Assignment\\Assignment3\\base_project\\References_code\\References_code\\results\\isic-2017_validation_predicted\\'
root_path2_validation_actual = 'C:\\NUS_ISS\\KE5108_DEVELOPING_INTELLIGENT_SYSTEMS_FOR_BUSINESS ANALYTICS\\Assignment\\Assignment3\\base_project\\References_code\\References_code\\datasets\\ISIC-2017_Validation_Part1_GroundTruth\\'

root_path_test_predicted = 'C:\\NUS_ISS\\KE5108_DEVELOPING_INTELLIGENT_SYSTEMS_FOR_BUSINESS ANALYTICS\\Assignment\\Assignment3\\base_project\\References_code\\References_code\\results\\ISIC-2017_Test_v2_Predicted\\'
root_path_test_actual = 'C:\\NUS_ISS\\KE5108_DEVELOPING_INTELLIGENT_SYSTEMS_FOR_BUSINESS ANALYTICS\\Assignment\\Assignment3\\base_project\\References_code\\References_code\\datasets\\ISIC-2017_Test_v2_Part1_GroundTruth\\'


extract_predicted_mask(root_path_validation_predicted+'model1\\',root_path2_validation_actual)

#img = load_img(root_path+'model1\\ISIC_0012095_segmentation.png')
#predicted_img_mask =img_to_array(img)
#predicted_img_mask2 = predicted_img_mask.flatten()
#predicted_img_mask2 = predicted_img_mask.reshape(-1)

#print(predicted_img_mask2.shape)

#img2 = load_img(root_path+'model1\\ISIC_0012095_segmentation.png')
#print(img2)
#actual_img_mask =img_to_array(img2)
#actual_img_mask2 = actual_img_mask.reshape(-1)
#actual_img_mask2 = actual_img_mask.flatten()
#print(actual_img_mask.shape)

#print(jaccard_similarity_score(actual_img_mask2, predicted_img_mask2))

