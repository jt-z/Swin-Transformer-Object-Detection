# 可视化VisDrone数据集的部分标注

import cv2
import random

def gen_colors(kinds = 12):

    colors = []
    for i in range(kinds):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color = (r, g, b)
        colors.append(color)
    return colors

order = True
def put_box(img, box_tupple, colors):
    start_point = (box_tupple[0], box_tupple[1])
    w,h = (box_tupple[2], box_tupple[3])
    score,category,truncation,occlusion =(box_tupple[4], box_tupple[5], box_tupple[6], box_tupple[7])
    end_point =(start_point[0]+w, start_point[1]+h)

    if category not in [11,3]:
        return

    # color =(255,0,0)
    color = colors[category]
    thickness = 1
    cv2.rectangle(img, start_point, end_point, color, thickness)
    cv2.putText(img,(f'S_TC: {score},{category},{truncation},{occlusion}'),  start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness, cv2.LINE_AA)
    # if order:
    #     cv2.putText(img, ('S_TC:'), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color,
    #                 thickness, cv2.LINE_AA)
    #     order = False


if __name__ == "__main__":
    img_path = '/Volumes/BiggerStore270G/datasets/VisDrone/VisDrone2019-DET-train/images/0000002_00005_d_0000014.jpg'
    img_path = '/Volumes/BiggerStore270G/datasets/VisDrone/VisDrone2019-DET-train/images/0000016_01352_d_0000069.jpg'
    img_path = '/Volumes/BiggerStore270G/datasets/VisDrone/VisDrone2019-DET-train/images/0000142_00858_d_0000027.jpg'

    img = cv2.imread(img_path)
    anno_name = img_path.split('/')[-1][:-4]
    anno_path = '/'.join(img_path.split('/')[:-2])+'/annotations/'+anno_name+'.txt'
    f_lines = open(anno_path,'r').readlines()
    ann_tuple = []
    for line in f_lines:
        line = line.replace('\n','').split(',')
        ann_tuple.append( [int(item) for item in line] )

    # print(ann_tuple)

    # 406,119,265,70,
    # ann_tuple = (406,119,265,70)


    #origin_ann = [109,110,5,8,1,10,0,1,              87,102,9,13,1,8,0,1,]
    flag = False
    if flag:
        origin_ann = [324,167,14,6,1,3,0,0,
                      57, 341, 11, 20, 1, 2, 0, 0,
                      137, 115, 6, 9, 1, 2, 0, 1,
                      109, 107, 4, 7, 1, 2, 0, 1,
                      223, 345, 8, 11, 1, 2, 0, 2
                      ]

    if flag :
        for i  in range(len(origin_ann)):
            temp_tuple.append(origin_ann[i])
            if len(temp_tuple) == 8:
                ann_tuple.append(temp_tuple)
                temp_tuple=[]

    # ann_tuple = [(708,471,74,33,1,4,0,1),(406,119,265,70,0,0,0,0) ]
    colors =gen_colors(12)
    for box in ann_tuple:
        put_box(img,box,colors)

    cv2.imshow('img', img)
    cv2.waitKey(0)