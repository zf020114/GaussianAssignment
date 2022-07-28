# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict
import os
import cv2
import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class AiTodDataset(CocoDataset):

    CLASSES = ('airplane', 'bridge', 'storage-tank', 'ship', 'swimming-pool', 'vehicle',
               'person', 'wind-mill')

    # def evaluate(self,
    #              results,
    #              metric='bbox',
    #              logger=None,
    #              jsonfile_prefix=None,
    #              classwise=True,
    #              proposal_nums=(100, 300, 1000),
    #              iou_thrs=None,
    #              metric_items = [
    #                     'mAP', 'mAP_50', 'mAP_75','mAP_vt', 'mAP_t', 'mAP_s', 'mAP_m', 'mAP_l'
    #                 ]):
    #     """Evaluation in COCO protocol.

    #     Args:
    #         results (list[list | tuple]): Testing results of the dataset.
    #         metric (str | list[str]): Metrics to be evaluated. Options are
    #             'bbox', 'segm', 'proposal', 'proposal_fast'.
    #         logger (logging.Logger | str | None): Logger used for printing
    #             related information during evaluation. Default: None.
    #         jsonfile_prefix (str | None): The prefix of json files. It includes
    #             the file path and the prefix of filename, e.g., "a/b/prefix".
    #             If not specified, a temp file will be created. Default: None.
    #         classwise (bool): Whether to evaluating the AP for each class.
    #         proposal_nums (Sequence[int]): Proposal number used for evaluating
    #             recalls, such as recall@100, recall@1000.
    #             Default: (100, 300, 1000).
    #         iou_thrs (Sequence[float], optional): IoU threshold used for
    #             evaluating recalls/mAPs. If set to a list, the average of all
    #             IoUs will also be computed. If not specified, [0.50, 0.55,
    #             0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
    #             Default: None.
    #         metric_items (list[str] | str, optional): Metric items that will
    #             be returned. If not specified, ``['AR@100', 'AR@300',
    #             'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
    #             used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
    #             'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
    #             ``metric=='bbox' or metric=='segm'``.

    #     Returns:
    #         dict[str, float]: COCO style evaluation metric.
    #     """

    #     metrics = metric if isinstance(metric, list) else [metric]
    #     allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
    #     for metric in metrics:
    #         if metric not in allowed_metrics:
    #             raise KeyError(f'metric {metric} is not supported')
    #     if iou_thrs is None:
    #         iou_thrs = np.linspace(
    #             .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    #     if metric_items is not None:
    #         if not isinstance(metric_items, list):
    #             metric_items = [metric_items]

    #     result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

    #     eval_results = OrderedDict()
    #     cocoGt = self.coco
    #     for metric in metrics:
    #         msg = f'Evaluating {metric}...'
    #         if logger is None:
    #             msg = '\n' + msg
    #         print_log(msg, logger=logger)

    #         if metric == 'proposal_fast':
    #             ar = self.fast_eval_recall(
    #                 results, proposal_nums, iou_thrs, logger='silent')
    #             log_msg = []
    #             for i, num in enumerate(proposal_nums):
    #                 eval_results[f'AR@{num}'] = ar[i]
    #                 log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
    #             log_msg = ''.join(log_msg)
    #             print_log(log_msg, logger=logger)
    #             continue

    #         iou_type = 'bbox' if metric == 'proposal' else metric
    #         if metric not in result_files:
    #             raise KeyError(f'{metric} is not in results')
    #         try:
    #             predictions = mmcv.load(result_files[metric])
    #             if iou_type == 'segm':
    #                 # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
    #                 # When evaluating mask AP, if the results contain bbox,
    #                 # cocoapi will use the box area instead of the mask area
    #                 # for calculating the instance area. Though the overall AP
    #                 # is not affected, this leads to different
    #                 # small/medium/large mask AP results.
    #                 for x in predictions:
    #                     x.pop('bbox')
    #                 warnings.simplefilter('once')
    #                 warnings.warn(
    #                     'The key "bbox" is deleted for more accurate mask AP '
    #                     'of small/medium/large instances since v2.12.0. This '
    #                     'does not change the overall mAP calculation.',
    #                     UserWarning)
    #             cocoDt = cocoGt.loadRes(predictions)
    #         except IndexError:
    #             print_log(
    #                 'The testing results of the whole dataset is empty.',
    #                 logger=logger,
    #                 level=logging.ERROR)
    #             break

    #         cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
    #         cocoEval.params.catIds = self.cat_ids
    #         cocoEval.params.imgIds = self.img_ids
    #         cocoEval.params.maxDets = list(proposal_nums)
    #         cocoEval.params.iouThrs = iou_thrs
    #         cocoEval.params.areaRng = [[0**2, 1e5**2], [0**2, 8**2], [8**2, 16**2], [16**2, 400**2]]
    #         cocoEval.params.areaRngLbl = ['all','small', 'medium', 'large']
            
    #         # mapping of cocoEval.stats
    #         coco_metric_names = {
    #             'mAP': 0,
    #             'mAP_50': 1,
    #             'mAP_75': 2,
    #             'mAP_vt': 3,
    #             'mAP_t': 4,
    #             'mAP_s': 5,
    #             'mAP_m': 6,
    #             'mAP_l': 7,
    #             'AR@100': 8,
    #             'AR@300': 9,
    #             'AR@1000': 10,
    #             'AR_s@1000': 11,
    #             'AR_m@1000': 12,
    #             'AR_l@1000': 13
    #         }
    #         if metric_items is not None:
    #             for metric_item in metric_items:
    #                 if metric_item not in coco_metric_names:
    #                     raise KeyError(
    #                         f'metric item {metric_item} is not supported')

    #         if metric == 'proposal':
    #             cocoEval.params.useCats = 0
    #             cocoEval.evaluate()
    #             cocoEval.accumulate()

    #             # Save coco summarize print information to logger
    #             redirect_string = io.StringIO()
    #             with contextlib.redirect_stdout(redirect_string):
    #                 cocoEval.summarize()
    #             print_log('\n' + redirect_string.getvalue(), logger=logger)

    #             if metric_items is None:
    #                 metric_items = [
    #                     'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
    #                     'AR_m@1000', 'AR_l@1000'
    #                 ]

    #             for item in metric_items:
    #                 val = float(
    #                     f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
    #                 eval_results[item] = val
    #         else:
    #             cocoEval.evaluate()
    #             cocoEval.accumulate()

    #             # Save coco summarize print information to logger
    #             redirect_string = io.StringIO()
    #             with contextlib.redirect_stdout(redirect_string):
    #                 cocoEval.summarize()
    #             print_log('\n' + redirect_string.getvalue(), logger=logger)
    #             # vtiny = cocoEval.summarize(1,
    #             #                   areaRng='vtiny',
    #             #                   maxDets=cocoEval.params.maxDets[2])
    #             # tiny = cocoEval.summarize()._summarize(1,
    #             #                   areaRng='tiny',
    #             #                   maxDets=self.params.maxDets[2])
    #             if classwise:  # Compute per-category AP
    #                 # Compute per-category AP
    #                 # from https://github.com/facebookresearch/detectron2/
    #                 precisions = cocoEval.eval['precision']
    #                 # precision: (iou, recall, cls, area range, max dets)
    #                 assert len(self.cat_ids) == precisions.shape[2]

    #                 results_per_category = []
    #                 for idx, catId in enumerate(self.cat_ids):
    #                     # area range index 0: all area ranges
    #                     # max dets index -1: typically 100 per image
    #                     nm = self.coco.loadCats(catId)[0]
    #                     precision = precisions[:, :, idx, 0, -1]
    #                     precision = precision[precision > -1]
    #                     if precision.size:
    #                         ap = np.mean(precision)
    #                     else:
    #                         ap = float('nan')
    #                     results_per_category.append(
    #                         (f'{nm["name"]}', f'{float(ap):0.3f}'))

    #                 num_columns = min(6, len(results_per_category) * 2)
    #                 results_flatten = list(
    #                     itertools.chain(*results_per_category))
    #                 headers = ['category', 'AP'] * (num_columns // 2)
    #                 results_2d = itertools.zip_longest(*[
    #                     results_flatten[i::num_columns]
    #                     for i in range(num_columns)
    #                 ])
    #                 table_data = [headers]
    #                 table_data += [result for result in results_2d]
    #                 table = AsciiTable(table_data)
    #                 print_log('\n' + table.table, logger=logger)

    #             if metric_items is None:
    #                 metric_items = [
    #                     'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
    #                 ]

    #             for metric_item in metric_items:
    #                 key = f'{metric}_{metric_item}'
    #                 val = float(
    #                     f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
    #                 )
    #                 eval_results[key] = val
    #             ap = cocoEval.stats[:6]
    #             eval_results[f'{metric}_mAP_copypaste'] = (
    #                 f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
    #                 f'{ap[4]:.3f} {ap[5]:.3f}')
    #     if tmp_dir is not None:
    #         tmp_dir.cleanup()
    #     return eval_results






#     def evaluate_rbox(self, results, work_dir=None, gt_dir=None):
#         dst_raw_path = osp.join(work_dir, 'results_before_nms')
#         dst_merge_path = osp.join(work_dir, 'results_after_nms')
#         if os.path.exists(dst_raw_path):
#             shutil.rmtree(dst_raw_path,True)
#         os.makedirs(dst_raw_path)
#         if os.path.exists(dst_merge_path):
#             shutil.rmtree(dst_merge_path,True)
#         os.makedirs(dst_merge_path)

#         imagesetfile=osp.join(osp.dirname(gt_dir), 'gt_list.txt')
#         generate_file_list(gt_dir,imagesetfile)

#         print('Saving results to {}'.format(dst_raw_path))
#         self.result_to_xml(results, os.path.join(work_dir,'pkl2xml'))
#         # self.xml2dota_txt(work_dir,dst_raw_path)
#         # self.result_to_txt(results, os.path.join(dst_raw_path,'result2txtdirect'))

#         # print('Merge results to {}'.format(dst_merge_path))
#         # mergebypoly(os.path.join(dst_raw_path,'result2txtdirect'), dst_merge_path)

#         # print('Start evaluation')
#         # detpath = osp.join(dst_merge_path, 'Task1_{:s}.txt')
#         # annopath = osp.join(gt_dir, '{:s}.txt')

#         # classaps = []
#         # map = 0
#         # for classname in self.CLASSES:
#         #     rec, prec, ap = voc_eval(detpath,
#         #                              annopath,
#         #                              imagesetfile,
#         #                              classname,
#         #                              ovthresh=0.5,
#         #                              use_07_metric=True)
#         #     map = map + ap
#         #     print(classname, ': ', ap)
#         #     classaps.append(ap)

#         # map = map / len(self.CLASSES)
#         # print('map:', map)
#         # classaps = 100 * np.array(classaps)
#         # print('classaps: ', classaps)
#         # # Saving results to disk
#         # with open(osp.join(work_dir, 'eval_results.txt'), 'w') as f: 
#         #     res_str = 'mAP:' + str(map) + '\n'
#         #     res_str += 'classaps: ' + ' '.join([str(x) for x in classaps])
#         #     f.write(res_str)
#         return map

#     def result_to_txt(self, results, results_path):
#         #这里需要更改 这里不生成xml 直接转换成txt
#         img_names = [img_info['filename'] for img_info in self.data_infos]
#         assert len(results) == len(img_names), 'len(results) != len(img_names)'
#         os.makedirs(results_path)
#         for classname in self.CLASSES:
#             f_out = open(osp.join(results_path, 'Task1_'+classname + '.txt'), 'w')
#             print(classname + '.txt')
#             # per result represent one image
#             for img_id, result in enumerate(results):
#                 for class_id, bboxes in enumerate(result):
#                     if self.CLASSES[class_id] != classname:
#                         continue

#                     if bboxes.size != 0:
#                         for bbox in bboxes:
#                             score= bbox[4]
#                             rbox_points =bbox[10:]
#                             temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
#                                 osp.splitext(img_names[img_id])[0], score, rbox_points[0], rbox_points[1],
#                                                                                                                         rbox_points[2], rbox_points[3],
#                                                                                                                         rbox_points[4], rbox_points[5],
#                                                                                                                         rbox_points[6], rbox_points[7])
#                             f_out.write(temp_txt)

#             f_out.close()

#     def result_to_xml(self, results, dst_path, score_threshold=0.03, nms_threshold=0.15,nms_maxnum=500 ):
#         CLASSES = self.CLASSES#dataset.CLASSESself.CLASSES#dataset.CLASSES
#         # img_names = [img_info['filename'] for img_info in self.img_infos]
#         # assert len(results) == len(img_names), 'len(results) != len(img_names)'
#         if not osp.exists(dst_path):
#             os.mkdir(dst_path)
#         for idx in range(len(self.img_ids)):
#             img_id = self.img_ids[idx]
#             img_name=self.data_infos[idx]['filename']
#             result = results[idx]
#             img_boxes=np.zeros((0,7))
#             for label in range(len(result)):
#                 bboxes = result[label]
#                 #过滤小阈值的目标
#                 keep= bboxes[:,4]>score_threshold
#                 bboxes=bboxes[keep]
#                 #这里开始写转换回来的函数
#                 if bboxes.shape[0]>0: 
#                     in_rbox=np.hstack((bboxes[...,5:10], bboxes[...,4:5],))
#                     rotateboxes,cv_rboxes=self.box2rotatexml(in_rbox,label)
#                     #rotate nms
#                     keep=nms_rotate_cpu(cv_rboxes,rotateboxes[:,5 ],nms_threshold, nms_maxnum)
#                     rotateboxes=rotateboxes[keep]
#                     img_boxes= np.vstack((img_boxes, rotateboxes))
#             write_rotate_xml(dst_path,img_name,[1024 ,1024,3],0.5,'0.5',img_boxes.reshape((-1,7)),CLASSES)

#     def box2rotatexml(self,bboxes,label):
#         rotateboxes=[]
#         cv_rboxes=[]
#         for i in range(bboxes.shape[0]):
#             if(bboxes.size != 0):
#                 # [xmin, ymin, xmax, ymax, score, x1, y1, x2, y2,x3,y3,x4,y4]=bboxes[i,:]
#                 [cx, cy, w,h,angle,score]=bboxes[i,:]
#                 # angle -= np.pi/2
#                 rotatebox=[cx, cy, w,h,angle,score,label]
#                 rotateboxes.append(rotatebox)
#                 cv_rboxes.append(rotate_rect2cv_np(rotatebox))
#         return np.array(rotateboxes), np.array(cv_rboxes)
   
#     def xml2dota_txt(self,dst_path, dst_raw_path):
#         CLASSES = self.CLASSES
#         NAME_LABEL_MAP={}
#         LABEl_NAME_MAP={}
#         for index, one_class in  enumerate(CLASSES):
#             NAME_LABEL_MAP[one_class]=index+1
#             LABEl_NAME_MAP[index+1]=one_class
#         file_paths = get_file_paths_recursive(dst_path, '.xml')
#         # Task2 # 建立写入句柄
#         write_handle_h = {}
#         for sub_class in CLASSES:
#             if sub_class == 'back_ground':
#                 continue
#             write_handle_h[sub_class] = open(os.path.join(dst_raw_path, 'Task1_%s.txt' % sub_class), 'a+')
#         # 循环写入
#         for count, xml_path in enumerate(file_paths):
#             img_size, gsd, imagesource, gtbox_label, extra =read_rotate_xml(xml_path,NAME_LABEL_MAP)
#             for i, rbox in enumerate(gtbox_label):
#                 rbox_cv=rotate_rect2cv(rbox)
#                 rect_box = cv2.boxPoints(rbox_cv)
#                 xmin,ymin,xmax,ymax=np.min(rect_box[:,0]),np.min(rect_box[:,1]),np.max(rect_box[:,0]),np.max(rect_box[:,1])
#                 # xmin,ymin,xmax,ymax,score=rbox[0:5]
#                 # command = '%s %.3f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n' % (os.path.splitext(os.path.split(xml_path)[1])[0],
#                 #                                             score,
#                 #                                             xmin, ymin, xmax, ymin,
#                 #                                             xmax, ymax, xmin, ymax,)
#                 command = '%s %.3f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n' % (os.path.splitext(os.path.split(xml_path)[1])[0],
#                                                                 np.float(extra[i]),
#                                                                 rect_box[0][0], rect_box[0][1], rect_box[1][0], rect_box[1][1],
#                                                                 rect_box[2][0], rect_box[2][1], rect_box[3][0], rect_box[3][1])
#                 write_handle_h[LABEl_NAME_MAP[rbox[5]]].write(command)
#         #关闭句柄
#         for sub_class in CLASSES:
#             if sub_class == 'back_ground':
#                 continue
#             write_handle_h[sub_class].close()


# def get_file_paths_recursive(folder=None, file_exts=None):
#     """ Get the absolute path of all files in given folder recursively
#     :param folder:
#     :param file_ext:
#     :return:
#     """
#     file_list = []
#     if folder is None:
#         return file_list
#     file_list = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(file_exts)]
#     return file_list

# def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):
#     keep = []#保留框的结果集合
#     order = scores.argsort()[::-1]#对检测结果得分进行降序排序
#     num = boxes.shape[0]#获取检测框的个数
#     suppressed = np.zeros((num), dtype=np.int)
#     for _i in range(num):
#         if len(keep) >= max_output_size:#若当前保留框集合中的个数大于max_output_size时，直接返回
#             break
#         i = order[_i]
#         if suppressed[i] == 1:#对于抑制的检测框直接跳过
#             continue
#         keep.append(i)#保留当前框的索引
#         # (midx,midy),(width,height), angle)
#         r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
#         #        r1 = ((boxes[i, 1], boxes[i, 0]), (boxes[i, 3], boxes[i, 2]), boxes[i, 4]) #根据box信息组合成opencv中的旋转bbox
#         #        print("r1:{}".format(r1))
#         area_r1 = boxes[i, 2] * boxes[i, 3]#计算当前检测框的面积
#         for _j in range(_i + 1, num):#对剩余的而进行遍历
#             j = order[_j]
#             if suppressed[i] == 1:
#                 continue
#             r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
#             area_r2 = boxes[j, 2] * boxes[j, 3]
#             inter = 0.0
#             int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]#求两个旋转矩形的交集，并返回相交的点集合
#             if int_pts is not None:
#                 order_pts = cv2.convexHull(int_pts, returnPoints=True)#求点集的凸边形
#                 int_area = cv2.contourArea(order_pts)#计算当前点集合组成的凸边形的面积
#                 inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 0.0000001)
#             if inter >= iou_threshold:#对大于设定阈值的检测框进行滤除
#                 suppressed[j] = 1
#     return np.array(keep, np.int64)

# def rotate_rect2cv_np(rotatebox):
#     #此程序将rotatexml中旋转矩形的表示，转换为cv2的RotateRect
#     [x_center,y_center,w,h,angle]=rotatebox[0:5]
#     angle_mod=angle*180/np.pi%180
#     if angle_mod>=0 and angle_mod<90:
#         [cv_w,cv_h,cv_angle]=[h,w,angle_mod-90]
#     if angle_mod>=90 and angle_mod<180:
#         [cv_w,cv_h,cv_angle]=[w,h,angle_mod-180]
#     cvbox=np.array([ x_center,y_center,cv_w,cv_h,cv_angle ])
#     return cvbox

# def rotate_rect2cv(rotatebox):
#     #此程序将rotatexml中旋转矩形的表示，转换为cv2的RotateRect
#     [x_center,y_center,w,h,angle]=rotatebox[0:5]
#     angle_mod=angle*180/np.pi%180
#     if angle_mod>=0 and angle_mod<90:
#         [cv_w,cv_h,cv_angle]=[h,w,angle_mod-90]
#     if angle_mod>=90 and angle_mod<180:
#         [cv_w,cv_h,cv_angle]=[w,h,angle_mod-180]
#     return ((x_center,y_center),(cv_w,cv_h),cv_angle)

# def read_rotate_xml(xml_path,NAME_LABEL_MAP):
#     """
#     :param xml_path: the path of voc xml
#     :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 5],
#             and has [xmin, ymin, xmax, ymax, label] in a per row
#     """
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#     img_width = None
#     img_height = None
#     box_list = []
#     extra=[]
#     for child_of_root in root:
#         if child_of_root.tag == 'folder':#读取gsd之前把它赋予到了folder字段
#             try:
#                 gsd = float(child_of_root.text)
#             except:
#                 gsd =0
#         if child_of_root.tag == 'size':
#             for child_item in child_of_root:
#                 if child_item.tag == 'width':
#                     img_width = int(child_item.text)
#                 if child_item.tag == 'height':
#                     img_height = int(child_item.text)
#                 if child_item.tag == 'depth':
#                     img_depth = 3#int(child_item.text)
#         if child_of_root.tag == 'source':
#             for child_item in child_of_root:
#                 if child_item.tag == 'database':
#                     imagesource=child_item.text
#         if child_of_root.tag == 'object':
#             label = None
#             for child_item in child_of_root:
#                 if child_item.tag == 'name':
#                     #TODO change
#                     #                    label_name=child_item.text.replace('plane','other').replace('\ufeffB-1B','B-1B').replace('F-31','F-35').replace('L-39','L-159')
#                     label_name=child_item.text.replace('\ufeff','')#.replace("其它","其他")#.replace('plane','bridge')#.replace('尼米兹级','航母').replace('圣安东尼奥','圣安东尼奥级').replace('圣安东尼奥级级','圣安东尼奥级')#.replace('塔瓦拉级','黄蜂级')
#                     label =NAME_LABEL_MAP[label_name]#float(child_item.text) #训练VOC用NAME_LABEL_MAP[child_item.text]#因为用自己的这边的ID是编号  训练卫星数据用1
#                 if child_item.tag == 'difficult':
#                     difficult=int(child_item.text)
#                 if child_item.tag == 'extra':
#                     extra.append(child_item.text)
#                 if child_item.tag == 'robndbox':
#                     tmp_box = [0, 0, 0, 0, 0,0,0]
#                     for node in child_item:
#                         if node.tag == 'cx':
#                             tmp_box[0] = float(node.text)
#                         if node.tag == 'cy':
#                             tmp_box[1] = float(node.text)
#                         if node.tag == 'w':
#                             tmp_box[2] = float(node.text)
#                         if node.tag == 'h':
#                             tmp_box[3] = float(node.text)
#                         if node.tag == 'angle':
#                             tmp_box[4] = float(node.text)
#                     assert label is not None, 'label is none, error'
#                     tmp_box[5]=label
#                     tmp_box[6]=difficult
#                     box_list.append(tmp_box)
#     #    gtbox_label = np.array(box_list, dtype=np.int32)
#     img_size=[img_height,img_width,img_depth]
#     return img_size,gsd,imagesource,box_list,extra

# ##添加写出为xml函数
# def write_rotate_xml(output_floder,img_name,size,gsd,imagesource,gtbox_label,CLASSES):#size,gsd,imagesource#将检测结果表示为中科星图比赛格式的程序,这里用folder字段记录gsd
#     voc_headstr = """\
#      <annotation>
#         <folder>{}</folder>
#         <filename>{}</filename>
#         <path>{}</path>
#         <source>
#             <database>{}</database>
#         </source>
#         <size>
#             <width>{}</width>
#             <height>{}</height>
#             <depth>{}</depth>
#         </size>
#         <segmented>0</segmented>
#         """
#     voc_rotate_objstr = """\
#        <object>
# 		<name>{}</name>
# 		<pose>Unspecified</pose>
# 		<truncated>0</truncated>
# 		<difficult>{}</difficult>
# 		<robndbox>
# 			<cx>{}</cx>
# 			<cy>{}</cy>
# 			<w>{}</w>
# 			<h>{}</h>
# 			<angle>{}</angle>
# 		</robndbox>
# 		<extra>{:.2f}</extra>
# 	</object>
#     """
#     voc_tailstr = '''\
#         </annotation>
#         '''
#     [floder,name]=os.path.split(img_name)
#     # filename=name.replace('.jpg','.xml')
#     filename=os.path.join(floder,os.path.splitext(name)[0]+'.xml')
#     foldername=os.path.split(img_name)[0]
#     head=voc_headstr.format(gsd,name,foldername,imagesource,size[1],size[0],size[2])
#     rotate_xml_name=os.path.join(output_floder,os.path.split(filename)[1])
#     f = open(rotate_xml_name, "w",encoding='utf-8')
#     f.write(head)
#     for i,box in enumerate (gtbox_label):
#         obj=voc_rotate_objstr.format(CLASSES[int(box[6])],0,box[0],box[1],box[2],box[3],box[4],box[5])
#         f.write(obj)
#     f.write(voc_tailstr)
#     f.close()

# def generate_file_list(img_dir,output_txt,file_ext='.txt'):
#     #读取原图路径
#     # img_dir=os.path.split(img_dir)[0]
#     imgs_path = get_file_paths_recursive(img_dir, file_ext)
#     f = open(output_txt, "w",encoding='utf-8')
#     for num,img_path in enumerate(imgs_path,0):
#         obj='{}\n'.format(os.path.splitext(os.path.split(img_path)[1])[0])
#         f.write(obj)
#     f.close()
#     print('Generate {} down!'.format(output_txt))