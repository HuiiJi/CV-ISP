# CV-ISP:NN based ISP for extreme CV-tasks 
<img src = "cvisp-background.png" width = "80%">

## **项目大纲**

- [Project](https://huiiji.github.io/CV-ISP/)

## **Denpendency**

git clone地址
```
git clone git@github.com:752413464/GAN_.py.git
cd GAN_
```
起一个虚拟环境，配置torch，torchvision等库（**以下命令需要先安装好Anaconda**）
```
conda create -n CV_ISP_demo python=3.7
conda activate CV_ISP_demo
conda install pytorch=1.7.0 torchvision=0.8.1 cudatoolkit=11.0 -c pytorch
```
配置好torch等框架，安装一些常用的cv视觉库。
```
pip install matplotlib scikit-image opencv-python tqdm 
```
或者可以直接安装requirements.txt
```
pip install -r requirements.txt
``` 
