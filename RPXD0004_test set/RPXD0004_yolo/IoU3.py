# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 17:58:41 2022

@author: Mingyu
"""

import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt


gt = []

os.chdir('./true_label')

true_label = os.listdir()

true_label.sort()

for i in true_label:
    
    f = open(i,'r')
    
    f = f.readlines()
    
    tmp = []
    
    for j in f:
        
        tmp_tmp = []
        
        j = j.replace('\n','')
        
        j = j.split()
        
        for k in j[1::]:
            
            tmp_tmp.append(eval(k))
        
        tmp.append(gt_bbox(tmp_tmp))
    
    gt.append(tmp)
        

os.chdir('..')
os.chdir('..')
os.chdir('..')

xx = []

yy = []

z = []

zz = []

zzz = []

check = []

for i in range(0,21):
        
        
    
    #确定检出的IoU阈值  
    
    i = i/20
    
    for j in range(0,21):
        
        #框子成立的阈值
        
        j = j/20
        
        xx.append(i)
        
        yy.append(j)
        
        tmp_arc = trans_yolo_json('RPXD0004_yolo',threshold=j)
        
        det_loc = tmp_arc[2]
        
        hits = []
        
        unhits = []
        
        gt_unhits = []
        
        for single_pic in range(len(det_loc)):
                        
            single_pic_gt_hits = []
            
            hit = 0
                
            unhit = 0
            
            ppp = len(det_loc[single_pic])
            
            for sing_pic_arc in det_loc[single_pic]:
                
                
                from copy import deepcopy
                
                old_hit = deepcopy(hit)
                
                
                
                for k in range(len(gt[single_pic])):
                        
                    if box_iou_xyxy(gt[single_pic][k],sing_pic_arc) >= i:
                        
                        hit += 1
                        
#                        single_pic_gt_hits.append(k)
                                                 
                        break


                # method 1 : all count begin


#                for k in range(len(gt[single_pic])):
#                        
#                    if box_iou_xyxy(gt[single_pic][k],sing_pic_arc) >= i:
#                        
#                        
#                        single_pic_gt_hits.append(k)
                                                 

                # method 1 : all count end





                # method 2 : max count begin

                tmp_k = []
                
                tmp_k_iou = []

                for k in range(len(gt[single_pic])):
                        
                    if box_iou_xyxy(gt[single_pic][k],sing_pic_arc) >= j:
                        
                        tmp_k.append(k)
                        
                        tmp_k_iou.append(box_iou_xyxy(gt[single_pic][k],sing_pic_arc))
                        

                if len(tmp_k_iou) != 0:
                    
                    single_pic_gt_hits.append(tmp_k[tmp_k_iou.index(max(tmp_k_iou))])


                # method 2 : max count end



                
                if old_hit == hit:
                
                    unhit += 1
                
                                        
            hits.append(hit)
            
            unhits.append(unhit)
            
            gt_unhits.append(len(gt[single_pic]) - len(set(single_pic_gt_hits)))
        
        
            if i == 0.85:
                
                check.append(single_pic_gt_hits)
        
        
    
        
        
        
        
        z.append(unhits)
        
        zz.append(hits)
        
        zzz.append(gt_unhits)


# Z : FP

# ZZ : TP

# ZZZ : FN


FP = np.array(z).sum(axis=1).reshape([21,21])

TP = np.array(zz).sum(axis=1).reshape([21,21])

FN = np.array(zzz).sum(axis=1).reshape([21,21])


FP = np.array(z).sum(axis=1)

TP = np.array(zz).sum(axis=1)

FN = np.array(zzz).sum(axis=1)

print(FP[353])

print(TP[353])

print(FN[353])

fp = np.array(FP[336:357])

tp = np.array(TP[336:357])

fn = np.array(FN[336:357])

tpr = tp/(tp+fn)

pre = tp/(tp+fp)

recall = tp/(tp+fn)

pre[-1] = 1

plt.scatter(fp,tpr)

plt.plot(fp,tpr)

plt.xlabel('FP')

plt.ylabel('TPR')

plt.show()


plt.scatter(fp/max(fp),tpr)

plt.plot(fp/max(fp),tpr)

plt.xlabel('norm_FP')

plt.ylabel('TPR')

plt.show()


plt.scatter(recall,pre)

plt.plot(recall,pre)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.show()


# auc for ROC


print(((fp[0:-1] - fp[1::])*tpr[0:-1]).sum())


print(((fp[0:-1] - fp[1::])*tpr[1::]).sum())


# auc for ROC_norm


print(((fp[0:-1]/69 - fp[1::]/69)*tpr[0:-1]).sum())


print(((fp[0:-1]/69 - fp[1::]/69)*tpr[1::]).sum())



# auc for ROC


print(((recall[0:-1] - recall[1::])*pre[0:-1]).sum())


print(((recall[0:-1] - recall[1::])*pre[1::]).sum())



# plot for mege fpr tpr

yyy = []

zzzz = []

for i in range(0,21):
    
    fp = np.array(FP[i*21:(i+1)*21])
    
    tp = np.array(TP[i*21:(i+1)*21])
    
    fn = np.array(FN[i*21:(i+1)*21])
    
    tpr = tp/(tp+fn)
    
    yyy += (fp/max(fp)).tolist()
    
    zzzz += tpr.tolist()




import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

import numpy as np

fig = plt.figure()

ax = plt.axes(projection='3d')


ax.scatter(xx, yyy, zzzz)

ax.set_xlabel("IoU threshold") #x轴上的名字

ax.set_ylabel("norm_FP") #y轴上的名字

ax.set_zlabel("TPR") #z轴上的名字


plt.show()


auc_x = []

high_z = []

low_z = []

for i in range(0,21):
    
    auc_x.append(i/20)
    
    fp = np.array(FP[i*21:(i+1)*21])
    
    tp = np.array(TP[i*21:(i+1)*21])
    
    fn = np.array(FN[i*21:(i+1)*21])
    
    tpr = tp/(tp+fn)
    


    high_z.append(((fp[0:-1]/max(fp) - fp[1::]/max(fp))*tpr[0:-1]).sum())


    low_z.append(((fp[0:-1]/max(fp) - fp[1::]/max(fp))*tpr[1::]).sum())



plt.plot(auc_x,high_z,label='Upper Bound of AUC')

plt.plot(auc_x,low_z,label='Lower Bound of AUC')


plt.xlabel('IoU threshold')

plt.ylabel('AUC with norm_FP')

plt.legend()

plt.show()



auc_x = []

high_z = []

low_z = []

for i in range(0,21):
    
    auc_x.append(i/20)
    
    fp = np.array(FP[i*21:(i+1)*21])
    
    tp = np.array(TP[i*21:(i+1)*21])
    
    fn = np.array(FN[i*21:(i+1)*21])
    
    tpr = tp/(tp+fn)
    


    high_z.append(((fp[0:-1] - fp[1::])*tpr[0:-1]).sum())


    low_z.append(((fp[0:-1] - fp[1::])*tpr[1::]).sum())



plt.plot(auc_x,high_z,label='Upper Bound of AUC')

plt.plot(auc_x,low_z,label='Lower Bound of AUC')


plt.xlabel('IoU threshold')

plt.ylabel('AUC with FP')

plt.legend()

plt.show()






auc_x = []

high_z = []

low_z = []

for i in range(0,21):
    
    auc_x.append(i/20)
    
    fp = np.array(FP[i*21:(i+1)*21])
    
    tp = np.array(TP[i*21:(i+1)*21])
    
    fn = np.array(FN[i*21:(i+1)*21])
    
    tpr = tp/(tp+fn)
    


    pre = tp/(tp+fp)
    
    recall = tp/(tp+fn)
    
    pre[-1] = 1

    high_z.append(((recall[0:-1] - recall[1::])*pre[0:-1]).sum())


    low_z.append(((recall[0:-1] - recall[1::])*pre[1::]).sum())



plt.plot(auc_x,high_z,label='Upper Bound of AUC')

plt.plot(auc_x,low_z,label='Lower Bound of AUC')


plt.xlabel('Mega\'s confidence threshold')

plt.ylabel('AP')

plt.legend()

plt.show()








