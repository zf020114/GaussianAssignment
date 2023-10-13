##Label Assignment Matters: A Gaussian Assignment Strategy for Tiny Object Detection
## RTSD dataset
Crop image dataset Download [Google Drive](https://drive.google.com/file/d/1qgGpduDC9WUHqQzX-cewtIfjLobYNcD0/view?usp=sharing) <br> whole image dataset [Baidu Drive](https://pan.baidu.com/s/1GV85qMMQyaY3qzij6WTmqQ?pwd=nudt) (Passwd:nudt)
## Trined model, logs and result file can be downloaded from the download link in the table.

|Model          |    AP     |    AP_vt  |  Speed | Download  | 
|:-------------:| :-------------: | :------------: | :----: | :---------------------------------------------------------------------------------------: |
| RetinaNet-S-GA         | 20.2     | 8.7   | 34.0 |   [Google Drive](https://drive.google.com/drive/folders/1valb_vfn9KW03ejbjbZcV6cPl_7aDAhD?usp=sharing) <br> [Baidu Drive](https://pan.baidu.com/s/1ILX82r9gJDrCmroMfWsXPw) (Passwd:nudt) |     
| FCOS-S-GA              | 19.6     | 7.9   | 34.8 |    [Google Drive](https://drive.google.com/drive/folders/1CdBPDm1PqVmV_apCbACYvt6c0GksDeb3?usp=sharing) <br> [Baidu Drive](https://pan.baidu.com/s/1fHLF7goL8cNvmKpkrTWICQ) (Passwd:nudt) |
| TTFNet-GA              | 21.8     | 10.3  | 34.3 |    [Google Drive](https://drive.google.com/drive/folders/1S2LurXTQ_v2RK6rq6_ecFNbRRJ_nTsxg?usp=sharing) <br> [Baidu Drive](https://pan.baidu.com/s/1YEXCNUjTuD9LNy8S9WbwEQ) (Passwd:nudt) |
| TTFNet-MiTB1-GA        |  24.2    | 10.4  | 37.6 |    [Google Drive](https://drive.google.com/drive/folders/1DgwdTGERFZnyOpVMiKEuOcjpnskggLw3?usp=sharing) <br> [Baidu Drive](https://pan.baidu.com/s/1Vq_5SYpWYJTWrYxZU_UpZg) (Passwd:nudt) |


## Install
Code (based on [mmdetection](https://github.com/open-mmlab/mmdetection)) 
The detailed installation steps are in the \docs\get_started.md
## Requirements

```
pytorch = 1.10.0
Linux or macOS (Windows is in experimental support)
Python 3.6+
PyTorch 1.3+
CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
GCC 5+
numpy = 1.21.2
mmcv-full >=1.3.17 
mmdet = 2.19.0
```
You can also use this command
```
pip install -r requirements.txt
```
1. Install mmcv-full.
```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```
```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```
2. Install MMDetection.
```shell
cd GuassionAssignment
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```
	
## How to use?

1) Download the [AI-TOD Dataset](https://drive.google.com/drive/folders/1mokzFtLCjyqalSEajYTUmyzXvOHAa4WX)
2) Install [mmdetection](https://github.com/open-mmlab/mmdetection)
3) Download our training models 
4) Edit the ```data_root, ``` in  config files in ```./configs_GA/```

👇 Core File 👇
>  Config file
>> config_GA/atss_darknet53_aitod_2x_ga.py.  
>> config_GA/fcos_darknet53_ga_aitod_2x.py.
>> config_GA/retina_darknet53_aitod_2x_ga.py  
>> config_GA/ttfnet_darknet53_aitod_iou_mask_ban_2x.py
>> config_GA/ttfnet_mitb1_aitod_160k_ctfocal2.py


## How to train?
```
python train.py ../config_GA/atss_darknet53_aitod_2x_ga.py 
```

## How to test?
```
python test.py ../config_GA/atss_darknet53_aitod_2x_ga.py ../{your_checkpoint_path} --eval bbox 
```

