## nvidia-docker(cuda+cudnn)+ocr_train

***

**host配置**

- CPU：8核至强Intel(R) Core(TM) i7-9700KF CPU @ 3.60GHz；
- 内存：32G内存(8*4)； 
- 硬盘：240GB SSD + 2TB机械盘 
- GPU：GTX-1060[Compute Capability:6.1]
- gcc: version 5.4.0
- kernel:4.15.0-142-generic
- NVIDIA driver: 465.31
- PaddlePallde: 如果你使用的是安培架构的GPU，推荐使用CUDA11.2。如果你使用的是非安培架构的GPU，推荐使用CUDA10.2，性能更优。
- CUDA：10.2
- tips：**Make sure you have installed the NVIDIA driver and Docker engine for your Linux distribution** **Note that you do not need to install the CUDA Toolkit on the host system, but the NVIDIA driver needs to be installed**

***
**CUDA-CUDNN-paddlepaddle-gpu版本一一对应——1**

- 官方文档1：https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/index_cn.html

**CUDA-CUDNN-paddlepaddle-gpu版本一一对应——2**

- 官方文档2：https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/windows-pip.html
- 如果你使用的是安培架构的GPU，推荐使用CUDA11.2。如果你使用的是非安培架构的GPU，推荐使用CUDA10.2，性能更优。2.2.2 CUDA10.2的PaddlePaddle：
1.安装paddlepaddle-gpu: python -m pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
***

**1.nvidia-docker安装**

- Setting up NVIDIA Container Toolkit：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit
  安装了nvidia-docker时已经安装了cuda,cudnn

- 设置`stable`存储库和 GPG 密钥:

  ```python
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
     && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
     && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
     
  OK
  deb https://nvidia.github.io/libnvidia-container/stable/ubuntu16.04/$(ARCH) /
  #deb https://nvidia.github.io/libnvidia-container/experimental/ubuntu16.04/$(ARCH) /
  deb https://nvidia.github.io/nvidia-container-runtime/stable/ubuntu16.04/$(ARCH) /
  #deb https://nvidia.github.io/nvidia-container-runtime/experimental/ubuntu16.04/$(ARCH) /
  deb https://nvidia.github.io/nvidia-docker/ubuntu16.04/$(ARCH) /
  ```

- `nvidia-docker2`更新包列表后安装包（和依赖项）：

  ```python
  sudo apt-get update
  ```

  ```python
  sudo apt-get install -y nvidia-docker2
  ```

- 设置好默认运行后重启Docker守护进程完成安装：

  ```python
  sudo systemctl restart docker
  ```

- 此时，可以通过运行基本 CUDA 容器来测试工作设置：

  ```python
  sudo docker run --rm --gpus all nvidia/cuda:10.2-base nvidia-smi
  ```

  ```python
  devops@bczx:~$ nvidia-docker images
  REPOSITORY    TAG         IMAGE ID       CREATED       SIZE
  nvidia/cuda   10.2-base   8a045058e34a   2 weeks ago   107MB
  ```

  此时已经通过上一步的测试自动pull了nvidia/cuda:10.2-base，至此，nvidia-docker已经安装成功

- 拉取预安装 PaddlePaddle 的镜像：

  ```python
  nvidia-docker pull registry.baidubce.com/paddlepaddle/paddle:2.1.0-gpu-cuda10.2-cudnn7
  ```

  直接把百度已经做好的images pull下来

- 用镜像构建并进入Docker容器：

  ```python
  nvidia-docker run --name paddle -it -v $PWD:/paddle registry.baidubce.com/paddlepaddle/paddle:2.1.0-gpu-cuda10.2-cudnn7 /bin/bash
  ```

  注：此时的container是可以使用cuda,cudnn的，因为我们使用的是nvidia-docker来生产的container的。

- 使用该contrainer运行ocr_train即可。

****

进入容器：

```python
λ b9b378727616
```

容器配置：

- 宿主机：nvidia-driver:465.31

- cuda: 10.2
- cudnn: 7
- paddlepaddle: paddlepaddle-gpu:2.1.0
- 挂载目录：$PWD:/paddle[将主目录挂载到/paddle中]

****

**2.cuda**

- ```python
  cat /usr/local/cuda/version.txt
  ```

**3.cudnn**

- ```python
  cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2
  ```

**4.导入ocr_train**

- 将host:/paddle[ocr_train]挂载到container:/paddle目录下

  ```python
  $PWD:/paddle 
  ```

- 生成并运行容器

  ```python
  nvidia-docker run --name paddle -it -v $PWD:/paddle registry.baidubce.com/paddlepaddle/paddle:2.1.0-gpu-cuda10.2-cudnn7 /bin/bash
  ```

- 至此可以在container:/paddle中进行所有的操作了

****

**镜像优化**

- 更改pip源
- 下载requirements.txt
- 重新构建镜像，push到hub

***

**visualDL**

- 命令行运行参数:--logdir；--host；--port

****

- scalar:

```python
with LogWriter(logdir="./log/scalar_test/train") as writer:
    for step in range(1000):
            # 向记录器添加一个tag为`acc`的数据
            writer.add_scalar(tag="acc", step=step, value=value[step])
            # 向记录器添加一个tag为`loss`的数据
            writer.add_scalar(tag="loss", step=step, value=1/(value[step] + 1))
```

```
visualdl --logdir ./log --port 8080
```

- 路径:

   **logdir**="./log/scalar_test/train"	# 在log文件夹下

  **--logdir** ./log					   # 定位到log文件夹

  **pwd:**	 ./log/scalar_test/train	 # 移花接木

  **log路径**:/paddle/ocr_train/visual/log/scalar_test/train/vdlrecords.1625049301.log

****

- image

```python
with LogWriter(logdir="./log/image_test/train") as writer:
    for step in range(6):
            # 添加一个图片数据
            writer.add_image(tag="eye",
                             img=random_crop("../../docs/images/eye.jpg"),
                             step=step)
```

```
visualdl --logdir ./log --port 8080
```

- 路径:

   **logdir**="./log/image_test/train"	# 在log文件夹下

  **--logdir** ./log					  # 定位到log文件夹

  **pwd:**	 ./log/image_test/train	 # 移花接木

  **log路径**:/paddle/ocr_train/visual/log/image_test/train/vdlrecords.1625049301.log

****



# train

- train.sh: python3 tools/train.py -c configs/rec/rec_icdar15_train.yml
- visualdl.sh: 
  - /paddle/ocr_train/output/rec/ic15**/vdl**

```python
cd output/rec/ic15	# 进入paddleocr默认设置的log路径下
visualdl --logdir ./vdl/ --host 0.0.0.0
```

此时即可随时观察train/eval的各个参数