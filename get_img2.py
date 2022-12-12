import os

import cv2

import glob

import shutil

import json

import numpy as np

import pandas as pd

from tqdm import tqdm

import subprocess

import traceback

import cv2

import math

from skimage.draw import polygon

from skimage.feature import peak_local_max


import numpy as np


def sync_execute(command: str) -> (str, str):
    try:
        subp = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        # 等待命令进程结束，超时时间2s，并触发异常
        resultcode = subp.wait(2)
        if resultcode == 0:
            output, err = subp.communicate()
            return output, err
        else:
            subp.kill()
            return '', 'subprcess killed'
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        return '', e



def make_dir(fp):
    """重新创建目录"""
    if os.path.exists(fp):
        
        shutil.rmtree(fp)
        
    os.makedirs(fp)


def video2img(fp_in):
    
    
    
    fp_out = 'video2pic_v5/' + fp_in.split('.')[0] + '/ori_pic'
    
    """
    :param fp_in: 输入视频path
    :param fp_out: 输出图片目录
    :return: 
    """
    
    file_name = fp_in.replace("\\", "/").split("/")[-1].split(".")[0]  # 视频文件名
    
    fp_in = os.path.abspath(fp_in)  # 绝对路径
    
    fp_out = os.path.abspath(fp_out)
    
    make_dir(fp_out)  # 重新创建输出目录

    # 打开 视频
    vc = cv2.VideoCapture(fp_in)
    
    frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    fps = vc.get(cv2.CAP_PROP_FPS)
    
    w, h = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 循环读取视频帧
    
#    counter = -1
    
    for i in tqdm(range(frames), desc="cut video"):
        
        
#        counter += 1
#        
#        if counter%1 == 0:
            
        rval, frame = vc.read()
        
        if rval:
            
            cv2.imwrite(f"""{fp_out}/{str(i).rjust(6, "0")}_test__{file_name}.png""", frame)  # 存储为图像，保存名为文件夹名
            
            cv2.waitKey(1)
            
        else:
            
            break
            
    
    vc.release()
    
    return fp_out






def mege_annotation(fp_in):
    
    global ori_pic
    
    global fp_out
    
    global kernel_path
    
    
    ori_pic = 'video2pic/' + fp_in.split('.')[0] + '/ori_pic'
    
    fp_out = 'video2pic/' + fp_in.split('.')[0] + '/mega_json'
    
    kernel_path = os.path.abspath('')
    
    
    file_name = fp_in.replace("\\", "/").split("/")[-1].split(".")[0]  # 视频文件名
    
    fp_in = os.path.abspath(fp_in)  # 绝对路径
    
    fp_out = os.path.abspath(fp_out)
    
    ori_pic = os.path.abspath(ori_pic)
    
    
    make_dir(fp_out)  # 重新创建输出目录
    
    
    os.chdir(ori_pic)
    
    pics = os.listdir()
    
    
    # 批量检测
    
#    os.system('export PYTHONPATH="/Users/yangmingyu/Desktop/new/cameratraps"&&export PYTHONPATH="$PYTHONPATH:/Users/yangmingyu/Desktop/new/ai4eutils"&&python ' + kernel_path + '/cameratraps/detection/run_detector_batch.py  ' + kernel_path +'/cameratraps/md_v4.1.0.pb ' + ori_pic + ' '  + fp_out + '/' + fp_in.split('.')[0] + '_mege_.json --output_relative_filenames --recursive')
    
    
#    os.chdir(kernel_path)
    
    
    
    
    # 单独检测
    
    for i in pics:
        
        
        os.system('export PYTHONPATH="/Users/yangmingyu/new/cameratraps"&&export PYTHONPATH="$PYTHONPATH:/Users/yangmingyu/new/ai4eutils"&&python ' + kernel_path + '/cameratraps/detection/run_detector_batch.py  ' + kernel_path +'/cameratraps/md_v4.1.0.pb ' + ori_pic + '/' + i + ' '  + fp_out + '/' + i.split('.')[0] + '.json')
        
        
        
    os.chdir(kernel_path)
    
    
    

def gt_bbox(box):  
    

#    x1cen,y1cen,width1,height1=0.792812,0.666667,0.295983,0.107843
    
    x1cen,y1cen,width1,height1=box[0],box[1],box[2],box[3]

    x1max=x1cen+(width1)/2
    
    x1min=x1cen-(width1)/2
    
    y1max=y1cen+(height1)/2
    
    y1min=y1cen-(height1)/2
    
    
#    print('%.4f'%x1min,'%.4f'%y1min,'%.4f'%x1max,'%.4f'%y1max)
    
    return [x1min, y1min, x1max, y1max]


def mege_bbox(box):
    
    
#    x2min,y2min,width2,height2=0.2004,0.6429,0.3119,0.09184
    
    x2min,y2min,width2,height2=box[0],box[1],box[2],box[3]

    
    x2max=x2min+width2
              
    y2max=y2min+height2
    
#    print('%.4f'%x2min,'%.4f'%y2min,'%.4f'%x2max,'%.4f'%y2max)
    
    return [x2min, y2min, x2max, y2max]




def trans_mege_json(fp_in,threshold=0.1):
        
    global json_files
    
    mege_json = 'video2pic/' + fp_in.split('.')[0] + '/mega_json'
    
    fp_out = 'video2pic/' + fp_in.split('.')[0] + '/mega_stats'
    
    
    kernel_path = os.path.abspath('')
        
    file_name = fp_in.replace("\\", "/").split("/")[-1].split(".")[0]  # 视频文件名
    
    fp_in = os.path.abspath(fp_in)  # 绝对路径
    
    fp_out = os.path.abspath(fp_out)
    
    make_dir(fp_out)
    
    mege_json = os.path.abspath(mege_json)

    os.chdir(mege_json)
    
    json_files = os.listdir()
    
    json_files.sort()
    
#    print(json_files)
    
    json_data = []
    
    for i in json_files:
        
        with open(i, "r") as read_file:
            
            data = json.load(read_file)
        
        json_data.append(dict(data))
    
    det_num = []
    
    det_loc= []
    
    det_cof = []
    
    
    for i in json_data:
        
        det = []
        
        cof = []
        
        tmp = i['images'][0]['detections']
        
        for j in tmp:
            
            if (j['category'] == '1') and (j['conf'] > threshold):
                
                det.append(mege_bbox(j['bbox']))
                
                cof.append(j['conf'])
        
        
        det_num.append(len(det))
        
        det_loc.append(det)
        
        det_cof.append(cof)
    
    
    os.chdir(fp_out)
    
    
    for i in range(len(det_loc)):
        
        np.savetxt(json_files[i].replace('.json','txt'),np.array(det_loc[i]))
        
    
    os.chdir(kernel_path)
    
    return [json_files, det_num, det_loc, det_cof]
    










def trans_yolo_json(fp_in,threshold=0.1):
        
    global json_files

    ori_pic = 'video2pic/' + fp_in.split('.')[0] + '/ori_pic'
    
    mege_json = 'video2pic/' + fp_in.split('.')[0] + '/yolo_json'
    
    fp_out = 'video2pic/' + fp_in.split('.')[0] + '/yolo_stats'
    
    
    kernel_path = os.path.abspath('')
        
    file_name = fp_in.replace("\\", "/").split("/")[-1].split(".")[0]  # 视频文件名
    
    fp_in = os.path.abspath(fp_in)  # 绝对路径
    
    fp_out = os.path.abspath(fp_out)
    
    make_dir(fp_out)

    ori_pic = os.path.abspath(ori_pic)

    mege_json = os.path.abspath(mege_json)


    os.chdir(ori_pic)

    ori_pic_file = os.listdir()

    ori_pic_file.sort()

    os.chdir(kernel_path)



    os.chdir(mege_json)


    
    json_files = os.listdir()
    
    json_files.sort()


    for i in ori_pic_file:

        if not(i.replace('png','txt') in json_files):

            f = open(i.replace('png','txt'),'w', encoding='utf-8')

            f.close()


    json_files = os.listdir()
    
    json_files.sort()

    
#    print(json_files)
    
    json_data = []
    
    for i in json_files:
        
#        print(i)
        
        tmp_file = open(i,'r')

        tmp_file = tmp_file.readlines()
        
        json_data.append(tmp_file)
    
    det_num = []
    
    det_loc= []
    
    det_cof = []
    
    
    for i in json_data:
        
        det = []
        
        cof = []
        
        
        for j in i:

            
            j = j.replace('\n','').split()

            if (j[0] == '0') and (float(j[-1]) > threshold):
                
                det.append(gt_bbox([float(j[1]),float(j[2]),float(j[3]),float(j[4])]))
                
                cof.append(float(j[-1]))
        
        
        det_num.append(len(det))
        
        det_loc.append(det)
        
        det_cof.append(cof)
    
    
    os.chdir(fp_out)
    
    
    for i in range(len(det_loc)):
        
        np.savetxt(json_files[i].replace('.txt','.txt'),np.array(det_loc[i]))
        
    
    os.chdir(kernel_path)
    
    return [json_files, det_num, det_loc, det_cof]














def draw_rectangle_by_point(img_file_path,new_img_file_path,points):
        
    image = cv2.imread(img_file_path)
    
    for item in points:
        
#        print("当前字符：",item)
        
        point=item[1]
        
        first_point=(int(point[0]),int(point[1]))
        
        last_point=(int(point[2]),int(point[3]))
        
        text_loc = (int((point[0]+(point[2]-point[0])/20)),int((point[1]+(point[3]-point[1])/10)))

        cv2.rectangle(image, first_point, last_point, (0, 255, 0), 1)#在图片上进行绘制框
        
        cv2.putText(image, item[0], text_loc, cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255,0,0), thickness=1)#在矩形框上方绘制该框的名称

    cv2.imwrite(new_img_file_path, image)         



def mege_vis(fp_in,thr):
    
    global pics
        
    tmp = trans_mege_json(fp_in,threshold=thr)
    
    det_loc = tmp[2]
    
    det_cof = tmp[3]
    
    
    ori_pic = 'video2pic/' + fp_in.split('.')[0] + '/ori_pic'

    mege_stat = 'video2pic/' + fp_in.split('.')[0] + '/mega_stats'
    
    fp_out = 'video2pic/' + fp_in.split('.')[0] + '/mega_pic'
    
    
    kernel_path = os.path.abspath('')
        
    file_name = fp_in.replace("\\", "/").split("/")[-1].split(".")[0]  # 视频文件名
    
    fp_in = os.path.abspath(fp_in)  # 绝对路径
    
    fp_out = os.path.abspath(fp_out)
    
    ori_pic = os.path.abspath(ori_pic)
    
    make_dir(fp_out)
    
    os.chdir(ori_pic)
    
    pics = os.listdir()
    
    pics.sort()
    
#    print(pics)
    
    
    for i in range(len(pics)):
        
        
    
#        try:
            
    
        img_file_path = os.path.abspath(pics[i])
            
        os.chdir(fp_out)
            
        new_img_file_path = os.path.abspath(pics[i])
            
        os.chdir(ori_pic)
            
        points = []
            
        image = cv2.imread(img_file_path)
            
        long = image.shape[1]
            
        short = image.shape[0]
            
            
#        if len(det_loc[i]) !=0:
                
        for j in range(len(det_loc[i])):
                    
                    
                    
            tmp_locc = det_loc[i][j]
                    
                    
                    
            locc = [tmp_locc[0]*long,tmp_locc[1]*short,tmp_locc[2]*long,tmp_locc[3]*short]
                    
    #                print(locc)
                    
            points.append((str(det_cof[i][j]),locc))
            
                
        draw_rectangle_by_point(img_file_path,new_img_file_path,points)
        
        
#        except:
            
#            continue
        
    
    os.chdir(kernel_path)

    return pics



# 计算IoU，矩形框的坐标形式为xyxy
def box_iou_xyxy(box1, box2):
    
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)
    
    # 相交坐标
    xmin = np.maximum(x1min, x2min)
    
    ymin = np.maximum(y1min, y2min)
    
    xmax = np.minimum(x1max, x2max)
    
    ymax = np.minimum(y1max, y2max)
    # 相交矩形行的高度、宽度、面积
    inter_h = np.maximum(ymax - ymin + 1., 0.)
    
    inter_w = np.maximum(xmax - xmin + 1., 0.)
    
    intersection = inter_h * inter_w
    # 计算相并面积
    union = s1 + s2 - intersection
    # 交并比
    iou = intersection / union
    
    return iou






def mega_video(fp_in):
    
    
    mege_pic = 'video2pic/' + fp_in.split('.')[0] + '/mega_pic'
    
    fp_out = 'video2pic/' + fp_in.split('.')[0] + '/mega_video'
    
    kernel_path = os.path.abspath('')
    
    file_name = fp_in.replace("\\", "/").split("/")[-1].split(".")[0]  # 视频文件名
    
    fp_in = os.path.abspath(fp_in)  # 绝对路径
    
    fp_out = os.path.abspath(fp_out)
    
    mege_pic = os.path.abspath(mege_pic)
    
    make_dir(fp_out)
    
    os.chdir(mege_pic)
    
    pics = os.listdir()
    
    pics.sort()
    
    
    imgs = []
    
    
    for i in pics:
        
        imgs.append(cv2.imread(i))
        
        
    os.chdir(fp_out)
    
    fps = 30 #视频每秒24帧
    
    size = (1920, 1080) #需要转为视频的图片的尺寸
    
    #可以使用cv2.resize()进行修改
    
    video = cv2.VideoWriter( file_name + ".avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    
    
    for i in imgs:
        
        video.write(i)
        
    video.release()
    
    cv2.destroyAllWindows()
    
    
    os.chdir(kernel_path)
    
    





def get_video(num_1, num_2):
    
    video_dir = 'D:/test/result.mp4'      # 输出视频的保存路径
    
    fps = 1      # 帧率
    
    img_size = (1920, 1080)      # 图片尺寸
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    
    for i in tqdm(range(num_1, num_2)):
        
        img_path = 'D:/test/tupian/' + '{}.jpg'.format(i)
        
        frame = cv2.imread(img_path)
        
        frame = cv2.resize(frame, img_size)  # 生成视频   图片尺寸和设定尺寸相同
        
        videoWriter.write(frame)  # 将图片写进视频里
        
    videoWriter.release()  # 释放资源







import cv2

import math

from skimage.draw import polygon

from skimage.feature import peak_local_max


import numpy as np

def polygon_IOU(polygon_1, polygon_2):
    
    """
    计算两个多边形的IOU
    :param polygon_1: [[row1, col1], [row2, col2], ...]
    :param polygon_2: 同上
    :return:
    """
    
    rr1, cc1 = polygon(polygon_2[:, 0], polygon_2[:, 1])
    
    rr2, cc2 = polygon(polygon_1[:, 0], polygon_1[:, 1])

    try:
        r_max = max(rr1.max(), rr2.max()) + 1
        
        c_max = max(cc1.max(), cc2.max()) + 1
    
    except:
        
        return 0

    canvas = np.zeros((r_max, c_max))
    
    canvas[rr1, cc1] += 1
    
    canvas[rr2, cc2] += 1
    
    union = np.sum(canvas > 0)
    
    if union == 0:
        
        return 0
    
    intersection = np.sum(canvas == 2)
    
    return intersection / union





class Compress_img:

    def __init__(self, img_path):
        self.img_path = img_path
        self.img_name = img_path.split('/')[-1]

    def compress_img_CV(self, loc, compress_rate=0.5, show=False):
        
        img = cv2.imread(self.img_path)
        
        heigh, width = img.shape[:2]
        # 双三次插值
        img_resize = cv2.resize(img, (int(width*compress_rate), int(heigh*compress_rate)),
                                interpolation=cv2.INTER_AREA)
        
        os.chdir(loc)
        
        
        files = loc + '/' + self.img_name.split('/')[-1]
        
        print(files)

        
        cv2.imwrite(files, img_resize)
        
        print("%s 已压缩，" % (self.img_name), "压缩率：", compress_rate)
        
        if show:
            
            cv2.imshow(self.img_name, img_resize)
            
            cv2.waitKey(0)



def compress(fp_in):
    
    
    ori_pic = 'video2pic/' + fp_in.split('.')[0] + '/ori_pic'
    
    fp_out = 'video2pic/' + fp_in.split('.')[0] + '/com_pic'
    
    kernel_path = os.path.abspath('')
    
    
    file_name = fp_in.replace("\\", "/").split("/")[-1].split(".")[0]  # 视频文件名
    
    fp_in = os.path.abspath(fp_in)
    
    fp_out = os.path.abspath(fp_out)
    
    ori_pic = os.path.abspath(ori_pic)
    
    make_dir(fp_out)

    os.chdir(ori_pic)
    
    pics = os.listdir()

    
    for i in pics:
        
        img_path = os.path.abspath(i)
    
        compress = Compress_img(img_path)
    
        # 使用opencv压缩图片
        compress.compress_img_CV(loc=fp_out)
        
        os.chdir(ori_pic)
    
    
    os.chdir(kernel_path)







def kcf_call(fp_in):
    
    ori_pic = 'video2pic/' + fp_in.split('.')[0] + '/ori_pic'
    
    mege_txt = 'video2pic/' + fp_in.split('.')[0] + '/mega_stats'
    
    fp_out = 'video2pic/' + fp_in.split('.')[0] + '/kcf_track'
        
    kernel_path = os.path.abspath('')
    
    kfc_script = os.path.abspath('11.py')
        
    file_name = fp_in.replace("\\", "/").split("/")[-1].split(".")[0]  # 视频文件名
    
    fp_in = os.path.abspath(fp_in)  # 绝对路径
    
    fp_out = os.path.abspath(fp_out)
    
    mege_txt = os.path.abspath(mege_txt)
    
    ori_pic = os.path.abspath(ori_pic)
    
    make_dir(fp_out)
    
    os.chdir(mege_txt)
    
    pics = os.listdir()

    pics.sort()
    
    for i in range(len(pics)):
                
        mege_pos = np.loadtxt(pics[i]).reshape([-1,4])
        
 #       for j in [0,1]:
        for j in range(mege_pos.shape[0]):
            
            if mege_pos.shape[0] == 1:
                
                tmp_mp = mege_pos.tolist()[0]
            
            else:
                
                tmp_mp = mege_pos[j].tolist()
                
            
            tmp_op = ori_pic
            
            tmp_ind = str(i)
            
            tmp_out = fp_out + '/' + pics[i].replace('.txt','') + '_' + str(j)
            
            os.system('/Users/yangmingyu/anaconda3/envs/py27/bin/python ' + kfc_script + ' -mp1 ' + str(tmp_mp[0]) + ' -mp2 ' + str(tmp_mp[1]) + ' -mp3 ' + str(tmp_mp[2]) + ' -mp4 ' + str(tmp_mp[3]) + ' -op ' + tmp_op + ' -ind ' + tmp_ind + ' -out ' + tmp_out)

#        break
    
    os.chdir(kernel_path)


def kcf_vis(fp_in,thr):
    
    global pics
        
    tmp = trans_mege_json(fp_in,threshold=thr)
    
    det_loc = tmp[2]
    
    det_cof = tmp[3]
    
    
    ori_pic = 'video2pic/' + fp_in.split('.')[0] + '/ori_pic'

    mege_stat = 'video2pic/' + fp_in.split('.')[0] + '/mega_stats'
    
    fp_out = 'video2pic/' + fp_in.split('.')[0] + '/mega_pic'
    
    
    kernel_path = os.path.abspath('')
        
    file_name = fp_in.replace("\\", "/").split("/")[-1].split(".")[0]  # 视频文件名
    
    fp_in = os.path.abspath(fp_in)  # 绝对路径
    
    fp_out = os.path.abspath(fp_out)
    
    ori_pic = os.path.abspath(ori_pic)
    
    make_dir(fp_out)
    
    os.chdir(ori_pic)
    
    pics = os.listdir()
    
    pics.sort()
    
#    print(pics)
    
    
    for i in range(len(pics)):
        
        
    
#        try:
            
    
        img_file_path = os.path.abspath(pics[i])
            
        os.chdir(fp_out)
            
        new_img_file_path = os.path.abspath(pics[i])
            
        os.chdir(ori_pic)
            
        points = []
            
        image = cv2.imread(img_file_path)
            
        long = image.shape[1]
            
        short = image.shape[0]
            
            
#        if len(det_loc[i]) !=0:
                
        for j in range(len(det_loc[i])):
                    
                    
                    
            tmp_locc = det_loc[i][j]
                    
                    
                    
            locc = [tmp_locc[0]*long,tmp_locc[1]*short,tmp_locc[2]*long,tmp_locc[3]*short]
                    
    #                print(locc)
                    
            points.append((str(det_cof[i][j]),locc))
            
                
        draw_rectangle_by_point(img_file_path,new_img_file_path,points)
        
        
#        except:
            
#            continue
        
    
    os.chdir(kernel_path)

    return pics












def KCF_draw_rectangle_by_point(img_file_path,new_img_file_path,points):
        
    image = cv2.imread(img_file_path)
    
    if len(points) == 1:
        
        for item in points:
            
    #        print("当前字符：",item)
            
            point=item[1]
            
            first_point=(int(point[0]),int(point[1]))
            
            last_point=(int(point[2]),int(point[3]))
            
            text_loc = (int((point[0]+(point[2]-point[0])/20)),int((point[1]+(point[3]-point[1])/10)))
    
            cv2.rectangle(image, first_point, last_point, (0, 0, 0), 1)#在图片上进行绘制框
            
            cv2.putText(image, item[0], text_loc, cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255,0,0), thickness=1)#在矩形框上方绘制该框的名称
    
        cv2.imwrite(new_img_file_path, image)     
    
    else:
        
        
        for item in points:
            
    #        print("当前字符：",item)
    
            indexx = points.index(item)
            
            if indexx == 0:
                
                
            
                point=item[1]
                
                first_point=(int(point[0]),int(point[1]))
                
                last_point=(int(point[2]),int(point[3]))
                
                text_loc = (int((point[0]+(point[2]-point[0])/20)),int((point[1]+(point[3]-point[1])/10)))
        
                cv2.rectangle(image, first_point, last_point, (0, 0, 0), 1)#在图片上进行绘制框
                
                cv2.putText(image, item[0], text_loc, cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255,0,0), thickness=1)#在矩形框上方绘制该框的名称

            else:
                
                point=item[1]
                
                first_point=(int(point[0]),int(point[1]))
                
                last_point=(int(point[2]),int(point[3]))
                
                text_loc = (int((point[0]+(point[2]-point[0])/20)),int((point[1]+(point[3]-point[1])/10)))
        
                cv2.rectangle(image, first_point, last_point, (255, 0, 0), 1)#在图片上进行绘制框
                
                cv2.putText(image, item[0], text_loc, cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255,0,0), thickness=1)#在矩形框上方绘制该框的名称

    
        cv2.imwrite(new_img_file_path, image)    

def kcf_mega_vis(fp_in):
    
    global pics
        
    # tmp = trans_mege_json(fp_in,threshold=thr)
    
    # det_loc = tmp[2]
    
    # det_cof = tmp[3]
    
    
    ori_pic = 'video2pic/' + fp_in.split('.')[0] + '/mega_pic'

    kcf_stat = 'video2pic/' + fp_in.split('.')[0] + '/kcf_track'
    
    fp_out = 'video2pic/' + fp_in.split('.')[0] + '/kcf_mege_pic'
    
    
    kernel_path = os.path.abspath('')
        
    file_name = fp_in.replace("\\", "/").split("/")[-1].split(".")[0]  # 视频文件名
    
    fp_in = os.path.abspath(fp_in)  # 绝对路径
    
    fp_out = os.path.abspath(fp_out)
    
    ori_pic = os.path.abspath(ori_pic)
    
    kcf_stat = os.path.abspath(kcf_stat)
    
    make_dir(fp_out)
    
    
    os.chdir(kcf_stat)
    
    vid = os.listdir()
    
    vid.sort()
    
    for j in vid:
        
        os.chdir(fp_out)
        
        make_dir(j)
        
        os.chdir(j)
        
        fp_out_out = os.path.abspath('')
    
        os.chdir(ori_pic)
        
        pics = os.listdir()
        
        pics.sort()
        
    #    print(pics)
        
        try:

            for i in range(len(pics)):
                
                
            
                    
            
                    img_file_path = os.path.abspath(pics[i])
                    
                    os.chdir(fp_out_out)
                    
                    new_img_file_path = os.path.abspath(pics[i])
                    
                    os.chdir(ori_pic)
                    
                    points = []
                    
                    image = cv2.imread(img_file_path)
                    
                    long = image.shape[1]
                    
                    short = image.shape[0]
                    
                    
            #        if len(det_loc[i]) !=0:
                        
                    # for j in range(len(det_loc[i])):
                    
                    os.chdir(kcf_stat)    
                    
                    os.chdir(j)
                    
                    tmp_file = os.listdir()
                    
                    tmp_file.sort()
                        
                    tmp_locc = np.loadtxt(tmp_file[i]).reshape([-1,])
                    
                    os.chdir(ori_pic) 
                    
                    
        #                print(locc)
                    
                    if i == 0:
                        
                        import copy
                    
                        locc = [tmp_locc[0]*long,tmp_locc[1]*short,tmp_locc[2]*long,tmp_locc[3]*short]
                        
                        s_locc = copy.deepcopy(locc)
        
                        points.append(('KCF_init',locc))
                    
                    else:
                        
                        points.append(('KCF_init',s_locc))
                        
                        locc = [tmp_locc[0]*long,tmp_locc[1]*short,tmp_locc[2]*long,tmp_locc[3]*short]
                                
                        points.append(('KCF',locc))
                    
                    KCF_draw_rectangle_by_point(img_file_path,new_img_file_path,points)

        except:
            
            continue
            
            
    #        else:
    #            
    #            draw_rectangle_by_point(img_file_path,new_img_file_path,points)
            
        
        
    
    os.chdir(kernel_path)





def kcf_mega_video(fp_in):
    
    
    mege_pic = 'video2pic/' + fp_in.split('.')[0] + '/kcf_mege_pic'
    
    fp_out = 'video2pic/' + fp_in.split('.')[0] + '/kcf_video'
        
    kernel_path = os.path.abspath('')
        
    file_name = fp_in.replace("\\", "/").split("/")[-1].split(".")[0]  # 视频文件名
    
    fp_in = os.path.abspath(fp_in)  # 绝对路径
    
    fp_out = os.path.abspath(fp_out)
    
    mege_pic = os.path.abspath(mege_pic)
    
    make_dir(fp_out)
    
    os.chdir(mege_pic)
    
    
    files = os.listdir()
    
    
    files.sort()
    
    
    for j in files:
        
        
        os.chdir(j)
    
        pics = os.listdir()
    
        pics.sort()
        
        
        imgs = []
        
        
        for i in pics:
            
            imgs.append(cv2.imread(i))
    
        
        os.chdir(fp_out)
        
        fps = 30 #视频每秒24帧
        
        size = (1920, 1080) #需要转为视频的图片的尺寸
    
        #可以使用cv2.resize()进行修改
    
        video = cv2.VideoWriter( j + ".avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    
    
        for i in imgs:
            
            video.write(i)
    
        video.release()
        
        cv2.destroyAllWindows()
    
        os.chdir(mege_pic)
    
    os.chdir(kernel_path)














def multi_mega_draw_rectangle_by_point(img_file_path,new_img_file_path,points,color=(0, 255, 0)):
        
    image = cv2.imread(img_file_path)
    
    for item in points:
        
#        print("当前字符：",item)
        
        point=item[1]
        
        first_point=(int(point[0]),int(point[1]))
        
        last_point=(int(point[2]),int(point[3]))
        
        text_loc = (int((point[0]+(point[2]-point[0])/20)),int((point[1]+(point[3]-point[1])/10)))

        cv2.rectangle(image, first_point, last_point, color, 1)#在图片上进行绘制框
        
        cv2.putText(image, item[0], text_loc, cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255,0,0), thickness=1)#在矩形框上方绘制该框的名称

    cv2.imwrite(new_img_file_path, image)         


def multi_mege_vis(fp_in,thr,file_dir1,file_dir2,cc):
    
    global pics
        
    tmp = trans_mege_json(fp_in,threshold=thr)
    
    det_loc = tmp[2]
    
    det_cof = tmp[3]
    
    
    ori_pic = 'video2pic/' + fp_in.split('.')[0] + '/' + file_dir1

    mege_stat = 'video2pic/' + fp_in.split('.')[0] + '/mega_stats'
    
    fp_out = 'video2pic/' + fp_in.split('.')[0] + '/' + file_dir2
    
    
    kernel_path = os.path.abspath('')
        
    file_name = fp_in.replace("\\", "/").split("/")[-1].split(".")[0]  # 视频文件名
    
    fp_in = os.path.abspath(fp_in)  # 绝对路径
    
    fp_out = os.path.abspath(fp_out)
    
    ori_pic = os.path.abspath(ori_pic)
    
    make_dir(fp_out)
    
    os.chdir(ori_pic)
    
    pics = os.listdir()
    
    pics.sort()
    
#    print(pics)
    
    
    for i in range(len(pics)):
        
        
    
#        try:
            
    
        img_file_path = os.path.abspath(pics[i])
        
        os.chdir(fp_out)
        
        new_img_file_path = os.path.abspath(pics[i])
        
        os.chdir(ori_pic)
        
        points = []
        
        image = cv2.imread(img_file_path)
        
        long = image.shape[1]
        
        short = image.shape[0]
        
        
#        if len(det_loc[i]) !=0:
            
        for j in range(len(det_loc[i])):
            
            
            
            tmp_locc = det_loc[i][j]
            
            
            
            locc = [tmp_locc[0]*long,tmp_locc[1]*short,tmp_locc[2]*long,tmp_locc[3]*short]
            
#                print(locc)
            
            points.append((str(det_cof[i][j]),locc))
    
        
        multi_mega_draw_rectangle_by_point(img_file_path,new_img_file_path,points,color=cc)
        
        
#        else:
#            
#            draw_rectangle_by_point(img_file_path,new_img_file_path,points)
            
        
        
    
    os.chdir(kernel_path)






def multi_mega_video(fp_in,file_dir2):
    
    
    mege_pic = 'video2pic/' + fp_in.split('.')[0] + '/' + file_dir2
    
    fp_out = 'video2pic/' + fp_in.split('.')[0] + '/multi_mega_video'
        
    kernel_path = os.path.abspath('')
        
    file_name = fp_in.replace("\\", "/").split("/")[-1].split(".")[0]  # 视频文件名
    
    fp_in = os.path.abspath(fp_in)  # 绝对路径
    
    fp_out = os.path.abspath(fp_out)
    
    mege_pic = os.path.abspath(mege_pic)
    
    make_dir(fp_out)
    
    os.chdir(mege_pic)
    
    pics = os.listdir()

    pics.sort()
    
    
    imgs = []
    
    
    for i in pics:
        
        imgs.append(cv2.imread(i))

    
    os.chdir(fp_out)
    
    fps = 30 #视频每秒24帧
    
    size = (1920, 1080) #需要转为视频的图片的尺寸

    #可以使用cv2.resize()进行修改

    video = cv2.VideoWriter( file_name + ".avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)


    for i in imgs:
        
        video.write(i)

    video.release()
    
    cv2.destroyAllWindows()
    
    
    os.chdir(kernel_path)













def kcf_test_vis(fp_in,thr):
    
    global pics
        
    tmp = trans_mege_json(fp_in,threshold=thr)
    
    det_loc = tmp[2]
    
    det_cof = tmp[3]
    
    
    ori_pic = 'video2pic/' + fp_in.split('.')[0] + '/ori_pic'

    mege_stat = 'video2pic/' + fp_in.split('.')[0] + '/mega_stats'
    
    fp_out = 'video2pic/' + fp_in.split('.')[0] + '/mega_pic'
    
    
    kernel_path = os.path.abspath('')
        
    file_name = fp_in.replace("\\", "/").split("/")[-1].split(".")[0]  # 视频文件名
    
    fp_in = os.path.abspath(fp_in)  # 绝对路径
    
    fp_out = os.path.abspath(fp_out)
    
    ori_pic = os.path.abspath(ori_pic)
    
    make_dir(fp_out)
    
    os.chdir(ori_pic)
    
    pics = os.listdir()
    
    pics.sort()
    
#    print(pics)
    
    
    for i in range(len(pics)):
        
        
    
#        try:
            
    
        img_file_path = os.path.abspath(pics[i])
            
        os.chdir(fp_out)
            
        new_img_file_path = os.path.abspath(pics[i])
            
        os.chdir(ori_pic)
            
        points = []
            
        image = cv2.imread(img_file_path)
            
        long = image.shape[1]
            
        short = image.shape[0]
            
            
#        if len(det_loc[i]) !=0:
                
        for j in range(len(det_loc[i])):
                    
                    
                    
            tmp_locc = det_loc[i][j]
                    
                    
                    
            locc = [tmp_locc[0]*long,tmp_locc[1]*short,tmp_locc[2]*long,tmp_locc[3]*short]
                    
    #                print(locc)
                    
            points.append((str(det_cof[i][j]),locc))
            
                
        draw_rectangle_by_point_(img_file_path,new_img_file_path,points)
        
        
#        except:
            
#            continue
        
    
    os.chdir(kernel_path)

    return pics



def draw_rectangle_by_point_(img_file_path,new_img_file_path,points):
        
    image = cv2.imread(img_file_path)
    
#    for item in points:
        
#        print("当前字符：",item)
        
#        point=item[1]
        
#        first_point=(int(point[0]),int(point[1]))
        
#        last_point=(int(point[2]),int(point[3]))
        
#        text_loc = (int((point[0]+(point[2]-point[0])/20)),int((point[1]+(point[3]-point[1])/10)))

#        cv2.rectangle(image, first_point, last_point, (0, 255, 0), 1)#在图片上进行绘制框
        
#        cv2.putText(image, item[0], text_loc, cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255,0,0), thickness=1)#在矩形框上方绘制该框的名称

    cv2.imwrite(new_img_file_path, image) 



if __name__ == "__main__":
    
    pass
    