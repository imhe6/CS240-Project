先前分出来了一个 `cs240` 分支略微修改了一些文件以解决一些小问题，目前建议各人从 `main` 分支专门分出一个个人分支来写，每次完成后合并到 `main` 中，避免合并冲突

## 关于配置环境

注：`pip install -r requirements.txt` 的时候会安装 `basicsr`，清华源 PyPI 镜像缺少它的依赖 `tb-nightly` 会导致安装失败，建议换其他镜像源

默认给的环境配置比较老：Python 3.8, CUDA 11.3 + ` torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 `

- 较新的 GPU 不支持这么低的 CUDA 版本，比如 Ada 架构（RTX 40系）最低只兼容到 CUDA 11.7/8

- **根据原文档，可以进行最低限度的升级**，例如 PyTorch 2.0.1 + CUDA 11.8

  - 在原文档 Create the Conda Environment 步骤：

    - 创建好 Python 3.8 环境之后，改为安装较新版本的包： `torch==2.0.1+cu118 torchaudio==2.0.2+cu118 torchvision==0.15.2+cu118`，经过测试能跑通，但会报很多 warning
    - 如果不想报 warning 则需要安装更低版本，比如 `torch1.13.1+cu117 torchaudio0.13.1+cu117 torchvision==0.14.1+cu117`（我没测试过）

  - 后续 Install Submodules and Other Dependencies 一节还需要本地编译 submodules 进行安装

    - 如果系统安装的 CUDA 编译工具版本差太多（比如 CUDA 13）会编译失败，建议用 conda 装一个低版本的，比如 CUDA 11.8

      ```bash
      # 安装 CUDA 11.8 的编译工具
      conda install -c "nvidia/label/cuda-11.8.0" cuda-compiler cuda-cudart-dev cuda-libraries-dev
      # 建议装一个 ninja 不然编译的时候会 complain
      conda install ninja
      ```

      之后需要先配置环境变量再编译安装 submodule：

      ```bash
      export CUDA_HOME=$CONDA_PREFIX
      export PATH=$CONDA_PREFIX/bin:$PATH
      
      # 按原文档的流程编译安装 submodules 和安装其他包
      cd SequenceMatters
      pip install submodules/diff-gaussian-rasterization
      pip install submodules/simple-knn
      pip install -r requirements.txt
      ```

- 也可以考虑用 Docker 容器，配置 GPU 直通到容器里作为 vGPU，这样就不受设备实际支持的 CUDA 版本影响了

## 关于 config

configs 目录下默认是有两个 config 文件 `blender.yml` 和 `mip360.yml`，默认分别对应两个数据集 `blender`(`nerf-synthetic`) 的训练过程（里面有些参数不一样，不能通用！）

以下对 config 文件里的参数简要说明

```yaml
# 指定端口，以防端口冲突导致无法进行
# 在 mip360.yml 原文件里是有这条的，但 blender.yml 原文件里没有，我自己加了一个上去
port: 6010

# 以下目录设置也支持相对位置（相对于CWD）
# 但是有时候不认，建议设为绝对位置

# 需要设置为实际存放 blender 数据集的目录
hr_source_dir: <path to directory of the HR dataset>

# 此目录用于 preprocess 步骤存放降低了分辨率的数据集
# （找不到目录会报错，需要提前手动建好）
lr_source_dir: <path to directory of the LR dataset to be saved>

# 此目录用于 render 步骤存放经过算法超分辨率后的数据集
vsr_save_dir: <path to directory of the VSR-upsampled dataset to be saved> 

# 用于 render 步骤存放演示视频 (可选）
# 原文件中为 Null，即不会产生演示视频
# 注意如果要存视频，路径需精确到文件名，比如 video/blender.mp4
video_save_path: Null

# 用于确定下面的 pretrained_models 的位置，没有改
vsr_model: psrt

# 需要手动设置为指向 spynet pretrained weights 的路径
# 原文件中是 Null，会指向 README.md 里说的（也是仓库里自带的） vsr/{vsr_model}/experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth
spynet_path: Null

# 此参数需要你手动设置为指向 vsr 模型 pretrained weights 的路径
# 原文件中是 Null，会指向 README.md 里说的（也是仓库里自带的）vsr/{vsr_model}/experiments/pretrained_models/PSRT_Vimeo.pth
# 另外原有的 README.md 提到的 PSRT pretrained model 下载地址里，除了仓库里本来就有的 PSRT_Vimeo 还有个 PSRT_REDS 模型，不知有什么区别
vsr_model_path: Null

```

除了上面这些，两个文件里还各有一些其他的参数，建议不要修改：

```yaml
# 以下是除了上面这些之外的参数
# 应当是特定于数据集的用于数据处理和训练的参数，建议不要修改

# blender.xml
downscale_factor: 4
subpixel: "bicubic"
white_background: false
als : false
num_images_in_sequence: 100
similarity: 'feature'
thres_values: [45]
lambda_tex: 0.60 

# mip360.xml
downscale_factor: 8
upscale_factor: 4
subpixel: "bicubic"
white_background: false
als : true
num_images_in_sequence: 8
similarity: 'pose'
thres_values: [30, 50]
lambda_tex: 0.40 
```

## 关于一键运行的脚本

`README.md` 原文档提到的两个一键运行的脚本 `run_blender.sh` `run_mip360.sh` ，原先比较简单，我重写了一下增加了其他一些功能，比如指定训练出来的模型的存放位置，详细的可以自己读一下脚本

**注意：** 原脚本每次运行都会先预处理数据集再进行其他过程，我为了节省时间注释掉了，现在**需要手动预处理数据集再运行对应脚本**，注意需要**先修改对应的 config 文件 (`.yml`) 指定数据集和预处理结果保存位置**：

```bash
# mip360 Preprocess (downscale datasets)
python scripts/downscale_dataset_mip360.py --config configs/mip360.yml

# blender Preprocess (downscale datasets)
python scripts/downscale_dataset_blender.py --config configs/blender.yml
```

## 关于数据集

根据 `README.md` 原文档，测试选取的数据集有 [NeRF Synthetic dataset](https://www.matthewtancik.com/nerf) (`blender`) 和 [Mip-NeRF 360 dataset](https://jonbarron.info/mipnerf360/) (`mip360`) 两个，分别解压即可

**注：** mip360 的 flowers 和 treehill 场景是不公开的（需要向数据集作者发申请说明使用意图才能获取到），因此我在对应的各个环节里把这两个场景去掉了

