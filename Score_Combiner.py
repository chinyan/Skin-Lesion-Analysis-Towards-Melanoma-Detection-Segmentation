

#combine 2 scores


result_unet = open('results.csv','r')
result_unet2 = open('results_vgg16_unet_on_600test.csv','r')
final_result = open('results_final.csv','w')

counter = 0
result_list = []
final_result.write('file_name,Jaccard_Score\n')
for r_unet in result_unet:
    if counter >0:
        r_unet_arr = r_unet.split(',')
        r_unet_tuple = r_unet_arr[0],r_unet_arr[1],r_unet_arr[2].replace('\n','')
        print(r_unet_tuple)
        result_list.append(r_unet_tuple)
    counter = counter+1

counter = 0
for r_unet2 in result_unet2:
    if counter > 0:
        r_unet2_arr = r_unet2.split(',')
        r_unet2_img = r_unet2_arr[0]
        r_unet2_jac = r_unet2_arr[1]
        top_score = r_unet2_jac
        for image_name,jaccard,dice in result_list:
            if image_name == r_unet2_img:
                if jaccard > top_score:
                    top_score = jaccard
                break
        final_result.write(r_unet2_img+','+top_score+'\n')
    counter = counter + 1

final_result.close()

