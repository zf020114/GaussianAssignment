import cv2
import mmcv
import numpy as np
import os
import cv2
# from mmdet.datasets.dota_k import DotaKDataset
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmcv import Config, DictAction

# configs='/media/zf/E/Dataset/2021ZKXT_aug_2/dardet_r50_DCN_fpn_2x_class10.py'
# pkl_path='/media/zf/E/Dataset/2021ZKXT_aug_2/workdir/DARDet_r50_DCN_2x_class10/test/test_class10_epoch13.pkl'
# dst_path = '/media/zf/E/Dataset/2021ZKXT_aug_2/workdir/DARDet_r50_DCN_2x_class10/test'

# configs='/media/zf/E/mmdetection219/workdir/ttfatss53_2x_800_aitod_ga/ttfatss_darknet53_aitod_2x_ga.py'
# pkl_path='/media/zf/E/mmdetection219/workdir/ttfatss53_2x_800_aitod_ga/result.pkl'
# dst_path = '/media/zf/E/mmdetection219/workdir/ttfatss53_2x_800_aitod_ga/xml'
configs='/media/zf/E/mmdetection219/workdir/ttfatss53_2x_800_aitod_ga/ttfatss_darknet53_aitod_2x_ga.py'
pkl_path='/media/zf/E/mmdetection219/workdir/ttfatss53_2x_800_aitod/result.pkl'
dst_path = '/media/zf/E/mmdetection219/workdir/ttfatss53_2x_800_aitod/xml'
show_score=0.3
# configs='/media/zf/E/Dataset/2021ZKXT_aug_2/dardet_r50_fpn_2x.py'
# pkl_path='/media/zf/E/Dataset/2021ZKXT_aug_2/workdir/DARDet_r50_2x/test_ISPRS/result_trainval_epoch12.pkl'
# work_dir = '/media/zf/E/Dataset/2021ZKXT_aug_2/workdir/DARDet_r50_2x'

def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):
    keep = []#保留框的结果集合
    order = scores.argsort()[::-1]#对检测结果得分进行降序排序
    num = boxes.shape[0]#获取检测框的个数
    suppressed = np.zeros((num), dtype=np.int)
    for _i in range(num):
        if len(keep) >= max_output_size:#若当前保留框集合中的个数大于max_output_size时，直接返回
            break
        i = order[_i]
        if suppressed[i] == 1:#对于抑制的检测框直接跳过
            continue
        keep.append(i)#保留当前框的索引
        # (midx,midy),(width,height), angle)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        #        r1 = ((boxes[i, 1], boxes[i, 0]), (boxes[i, 3], boxes[i, 2]), boxes[i, 4]) #根据box信息组合成opencv中的旋转bbox
        #        print("r1:{}".format(r1))
        area_r1 = boxes[i, 2] * boxes[i, 3]#计算当前检测框的面积
        for _j in range(_i + 1, num):#对剩余的而进行遍历
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0
            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]#求两个旋转矩形的交集，并返回相交的点集合
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)#求点集的凸边形
                int_area = cv2.contourArea(order_pts)#计算当前点集合组成的凸边形的面积
                inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 0.0000001)
            if inter >= iou_threshold:#对大于设定阈值的检测框进行滤除
                suppressed[j] = 1
    return np.array(keep, np.int64)

def box2rotatexml(bboxes,label):
    rotateboxes=[]
    cv_rboxes=[]
    for i in range(bboxes.shape[0]):
        if(bboxes.size != 0):
            # [xmin, ymin, xmax, ymax, score, x1, y1, x2, y2,x3,y3,x4,y4]=bboxes[i,:]
            [cx, cy, w,h,angle,score]=bboxes[i,:]
            # angle -= np.pi/2
            rotatebox=[cx, cy, w,h,angle,score,label]
            rotateboxes.append(rotatebox)
            cv_rboxes.append(rotate_rect2cv_np(rotatebox))
    return np.array(rotateboxes), np.array(cv_rboxes)

def rotate_rect2cv_np(rotatebox):
    #此程序将rotatexml中旋转矩形的表示，转换为cv2的RotateRect
    [x_center,y_center,w,h,angle]=rotatebox[0:5]
    angle_mod=angle*180/np.pi%180
    if angle_mod>=0 and angle_mod<90:
        [cv_w,cv_h,cv_angle]=[h,w,angle_mod-90]
    if angle_mod>=90 and angle_mod<180:
        [cv_w,cv_h,cv_angle]=[w,h,angle_mod-180]
    cvbox=np.array([ x_center,y_center,cv_w,cv_h,cv_angle ])
    return cvbox

##添加写出为xml函数
def write_rotate_xml(output_floder,img_name,size,gsd,imagesource,gtbox_label,CLASSES):#size,gsd,imagesource#将检测结果表示为中科星图比赛格式的程序,这里用folder字段记录gsd
    voc_headstr = """\
     <annotation>
        <folder>{}</folder>
        <filename>{}</filename>
        <path>{}</path>
        <source>
            <database>{}</database>
        </source>
        <size>
            <width>{}</width>
            <height>{}</height>
            <depth>{}</depth>
        </size>
        <segmented>0</segmented>
        """
    voc_rotate_objstr = """\
       <object>
		<name>{}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>{}</difficult>
		<robndbox>
			<cx>{}</cx>
			<cy>{}</cy>
			<w>{}</w>
			<h>{}</h>
			<angle>{}</angle>
		</robndbox>
		<extra>{:.2f}</extra>
	</object>
    """
    voc_tailstr = '''\
        </annotation>
        '''
    [floder,name]=os.path.split(img_name)
    # filename=name.replace('.jpg','.xml')
    filename=os.path.join(floder,os.path.splitext(name)[0]+'.xml')
    foldername=os.path.split(img_name)[0]
    head=voc_headstr.format(gsd,name,foldername,imagesource,size[1],size[0],size[2])
    rotate_xml_name=os.path.join(output_floder,os.path.split(filename)[1])
    f = open(rotate_xml_name, "w",encoding='utf-8')
    f.write(head)
    for i,box in enumerate (gtbox_label):
        obj=voc_rotate_objstr.format(CLASSES[int(box[6])],0,box[0],box[1],box[2],box[3],box[4],box[5])
        f.write(obj)
    f.write(voc_tailstr)
    f.close()
    
def result_to_xml( dataset, results, dst_path, score_threshold=0.03, nms_threshold=0.65,nms_maxnum=300 ):
    CLASSES = dataset.CLASSES#dataset.CLASSESself.CLASSES#dataset.CLASSES
    # img_names = [img_info['filename'] for img_info in self.img_infos]
    # assert len(results) == len(img_names), 'len(results) != len(img_names)'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    for idx in range(len(dataset.img_ids)):
        # img_id = dataset.img_ids[idx]
        img_name=dataset.data_infos[idx]['filename']
        result = results[idx]
        img_boxes=np.zeros((0,7))
        for label in range(len(result)):
            bboxes = result[label]
            #过滤小阈值的目标
            keep= bboxes[:,4]>score_threshold
            bboxes=bboxes[keep]
            #这里开始写转换回来的函数
            if bboxes.shape[0]>0: 
                # in_rbox=np.hstack((bboxes[...,5:10], bboxes[...,4:5],))
                cx,cy,w,h=(bboxes[...,0:1]+bboxes[...,2:3])/2, (bboxes[...,1:2]+bboxes[...,3:4])/2,(bboxes[...,2:3]-bboxes[...,0:1]),(bboxes[...,3:4]-bboxes[...,1:2])
                angle =np.zeros_like(cx)
                in_rbox=np.hstack((cx,cy,w,h,angle,bboxes[...,4:5]))
                rotateboxes,cv_rboxes=box2rotatexml(in_rbox,label)
                #rotate nms
                keep=nms_rotate_cpu(cv_rboxes,rotateboxes[:,5 ],nms_threshold, nms_maxnum)
                rotateboxes=rotateboxes[keep]
                img_boxes= np.vstack((img_boxes, rotateboxes))
        write_rotate_xml(dst_path,img_name,[1024 ,1024,3],0.5,'0.5',img_boxes.reshape((-1,7)),CLASSES)

def pkl2xml(configs,pkl_path,dst_path,show_score):
    # dataset=DotaKDataset
    cfg = Config.fromfile(configs)
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    dataset = build_dataset(cfg.data.test)
    results=mmcv.load(pkl_path)
    result_to_xml(dataset,results, dst_path,show_score )
    # dataset.evaluate_rbox(results, work_dir, gt_dir)
    
pkl2xml(configs,pkl_path,dst_path,show_score)