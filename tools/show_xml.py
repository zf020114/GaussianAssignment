# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:06:37 2019

@author: admin
"""
import numpy as np
import os
import cv2
from timeit import default_timer as timer
import xml.etree.ElementTree as ET
import xmltools
import pkl2xml

def get_file_paths_recursive(folder=None, file_exts=None):
    """ Get the absolute path of all files in given folder recursively
    :param folder:
    :param file_ext:
    :return:
    """
    file_list = []
    if folder is None:
        return file_list
    file_list = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(file_exts)]
    return file_list

def get_label_name_map(NAME_LABEL_MAP):
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

def read_rotate_xml(xml_path,NAME_LABEL_MAP):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 5],
           and has [xmin, ymin, xmax, ymax, label] in a per row
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    extra=[]
    for child_of_root in root:
        if child_of_root.tag == 'folder':#读取gsd之前把它赋予到了folder字段
            try:
                gsd = float(child_of_root.text)
            except:
                gsd =0
        if child_of_root.tag == 'gsd':
            gsd = float(child_of_root.text)
        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)
                if child_item.tag == 'depth':
                    img_depth = 3#int(child_item.text)
        
        if child_of_root.tag == 'source':
            for child_item in child_of_root:
                if child_item.tag == 'database':
                    imagesource=child_item.text
        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    #TODO change
#                    label_name=child_item.text.replace('plane','other').replace('\ufeffB-1B','B-1B').replace('F-31','F-35').replace('L-39','L-159')
                    label_name=child_item.text.replace('\ufeff','').replace("其它","其他")#.replace('plane','bridge')#.replace('尼米兹级','航母').replace('圣安东尼奥','圣安东尼奥级').replace('圣安东尼奥级级','圣安东尼奥级')#.replace('塔瓦拉级','黄蜂级')
                    label =NAME_LABEL_MAP[label_name]#float(child_item.text) #训练VOC用NAME_LABEL_MAP[child_item.text]#因为用自己的这边的ID是编号  训练卫星数据用1
                if child_item.tag == 'difficult':
                    difficult=int(child_item.text)
                if child_item.tag == 'extra':
                    extra.append(child_item.text)
                if child_item.tag == 'robndbox':
                    tmp_box = [0, 0, 0, 0, 0,0,0]
                    for node in child_item:
                        if node.tag == 'cx':
                            tmp_box[0] = float(node.text)
                        if node.tag == 'cy':
                            tmp_box[1] = float(node.text)
                        if node.tag == 'w':
                            tmp_box[2] = float(node.text)
                        if node.tag == 'h':
                            tmp_box[3] = float(node.text)
                        if node.tag == 'angle':
                            tmp_box[4] = float(node.text)
                    assert label is not None, 'label is none, error'
                    tmp_box[5]=label
                    tmp_box[6]=difficult
                    box_list.append(tmp_box)
#    gtbox_label = np.array(box_list, dtype=np.int32) 
    img_size=[img_height,img_width,img_depth]
    return img_size,gsd,imagesource,box_list,extra

color_list = np.array(
        [
            1.0, 0, 0,
            0, 1.0,0,
            0, 0, 1.0,
        ]
    ).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
colors = [(color_list[_]).astype(np.uint8) \
            for _ in range(len(color_list))]

def read_rec_to_rot(xml_file,NAME_LABEL_MAP):
    """
    读取xml文件,得到【cx,cy,w,h,angle】
    将所有原本的正框标注修改为斜框标注
    原有的斜框标注保持不变
    """    
    tree=xmltools.read_xml(xml_file)
    object_list=xmltools.find_nodes(tree,"object")
    total_object = []
    for obj in object_list:
        for attr in obj:
            if attr.tag=='name':
                label=NAME_LABEL_MAP[attr.text]
            if attr.tag=='bndbox':
                xmin=float(xmltools.find_nodes(attr,"xmin")[0].text)
                ymin=float(xmltools.find_nodes(attr,"ymin")[0].text)
                xmax=float(xmltools.find_nodes(attr,"xmax")[0].text)
                ymax=float(xmltools.find_nodes(attr,"ymax")[0].text)
                cx=(xmin+xmax)/2
                cy=(ymin+ymax)/2
                w=xmax-xmin
                h=ymax-ymin
                angle=0
                
                text1 = [cx,cy,w,h,angle,label,0]
                total_object.append(text1)
            if attr.tag=='robndbox':
                cx=float(xmltools.find_nodes(attr,"cx")[0].text)
                cy=float(xmltools.find_nodes(attr,"cy")[0].text)
                w=float(xmltools.find_nodes(attr,"w")[0].text)
                h=float(xmltools.find_nodes(attr,"h")[0].text)
                angle=float(xmltools.find_nodes(attr,"angle")[0].text)
                text1 = [cx,cy,w,h,angle,label,0]
                total_object.append(text1)
    return total_object
 
def show_rotate_box(src_img,rotateboxes, c):  
    cx, cy, w,h,Angle=rotateboxes[:,0], rotateboxes[:,1], rotateboxes[:,2], rotateboxes[:,3], rotateboxes[:,4]
    p_rotate=[]
    for i in range(rotateboxes.shape[0]):
        RotateMatrix=np.array([
                              [np.cos(Angle[i]),-np.sin(Angle[i])],
                              [np.sin(Angle[i]),np.cos(Angle[i])]])

        rhead,r1,r2,r3,r4=np.transpose([0,-h[i]/2]),np.transpose([-w[i]/2,-h[i]/2]),np.transpose([w[i]/2,-h[i]/2]),np.transpose([w[i]/2,h[i]/2]),np.transpose([-w[i]/2,h[i]/2])
        # rhead=np.transpose(np.dot(RotateMatrix, rhead))+[cx[i],cy[i]]
        p1=np.transpose(np.dot(RotateMatrix, r1))+[cx[i],cy[i]]
        p2=np.transpose(np.dot(RotateMatrix, r2))+[cx[i],cy[i]]
        p3=np.transpose(np.dot(RotateMatrix, r3))+[cx[i],cy[i]]
        p4=np.transpose(np.dot(RotateMatrix, r4))+[cx[i],cy[i]]
        p_rotate_=np.int32(np.vstack((p1,p2,p3,p4)))
        cv2.polylines( src_img, [np.array(p_rotate_)], True, c[int(rotateboxes[i,5])].tolist(), 2)
    return src_img


def main():
    # input_img_dir = '/media/zf/E/Dataset/AI-TOD/show_example'
    # file_ext='.png'
    # input_xml_dir ='/media/zf/E/Dataset/AI-TOD/val/atss_single_ga_xml'
    # input_gt_xml_dir = '/media/zf/E/Dataset/AI-TOD/annotations/val2017xml'
    # outputfolder='/media/zf/E/Dataset/AI-TOD/val/atss_single_ga_xml/show_draw'

    input_img_dir = '/media/zf/E/Dataset/AI-TOD/show_example'
    file_ext='.png'
    input_xml_dir ='/media/zf/E/Dataset/AI-TOD/val/atss_single_xml'
    input_gt_xml_dir = '/media/zf/E/Dataset/AI-TOD/annotations/val2017xml'
    outputfolder='/media/zf/E/Dataset/AI-TOD/val/atss_single_xml/show_draw'
    
    NAME_LABEL_MAP ={
            'airplane': 1,
            'bridge': 2,
            'storage-tank': 3,
            'ship': 4,
            'swimming-pool': 5,
            'vehicle': 6,
            ' person': 7,
            'wind-mill': 8
            }
    #新建输出文件夹
    if not os.path.isdir(outputfolder):
        os.makedirs(outputfolder)

    #读取原图全路径  
    imgs_path = get_file_paths_recursive(input_img_dir, file_ext) 
    #旋转角的大小，整数表示逆时针旋转
    imgs_total_num=len(imgs_path)
    for num,img_path in enumerate(imgs_path,0):
        start = timer()
        img = cv2.imread(img_path,-1)
        xml_path=img_path.replace(file_ext,'.xml').replace(input_img_dir,input_xml_dir)
        img_size,gsd,imagesource,detbox_label,extra=read_rotate_xml(xml_path,NAME_LABEL_MAP)
        detbox_label = np.array(detbox_label)
        detbox_label[:,6]=0  #将所有检测到的目标置信度设置为0
        detbox_score = np.hstack((detbox_label[:,0:5],detbox_label[:,6:7]))
       
        gt_path = img_path.replace(file_ext,'.xml').replace(input_img_dir,input_gt_xml_dir)
        gtbox_label=np.array(read_rec_to_rot(gt_path,NAME_LABEL_MAP))
        gtbox_label[:,6]=1 #将所有GT标置信度设置为1
        gtbox_score = np.hstack((gtbox_label[:,0:5],gtbox_label[:,6:7]))
        nms_array1 =  np.vstack((detbox_score,gtbox_score))
        rotateboxes1,cv_rboxes1=pkl2xml.box2rotatexml(nms_array1,1)
        #rotate nms
        keep=pkl2xml.nms_rotate_cpu(cv_rboxes1,rotateboxes1[:,5 ],0.5, 600)
        rotateboxes1=rotateboxes1[keep]
        img=show_rotate_box(img,np.array(rotateboxes1) , color_list )
        #经过nms后 虚警 FP的分数是0   正确 和漏检  (TP + FN)的的分数是1
        
        
        #接下来画漏检 
        gtbox_score[:,5]=2 #将GT的置信度设置为2
        detbox_score[:,5]=3 #将所有检测到的目标置信度设置为3
        nms_array2 = np.vstack((detbox_score,gtbox_score))
        rotateboxes2,cv_rboxes2=pkl2xml.box2rotatexml(nms_array2,1)
        keep2=pkl2xml.nms_rotate_cpu(cv_rboxes2,rotateboxes2[:,5 ],0.5, 600)
        rotateboxes2=rotateboxes2[keep2]
        #经过nms后 漏检 FN的分数是2   正确 和虚警  (TP + FP)的的分数是3
        keep3 = rotateboxes2[:,5]==2
        rotateboxes2 = rotateboxes2[keep3]
        #上面画完后,我只需要继续覆盖画出漏检把漏检的颜色变过来 漏检是红色
        img=show_rotate_box(img,np.array(rotateboxes2) , color_list )
        
        img_name=os.path.splitext(os.path.split(img_path)[1])[0]+'.jpg'
        jpg_img_path=os.path.join(outputfolder,img_name)
        cv2.imwrite(jpg_img_path,img)
        

if __name__ == '__main__':
    main()