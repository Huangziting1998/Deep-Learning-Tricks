# Deep Learning Tricks



## 模型效果差的原因

```
（1）模型自身结构->模型的表达能力（深度和宽度）

（2）超参数选择 -学习率，优化器，学习率调整策略

（3）数据模型不匹配 

（4）数据集构造：没有足够数据、分类不均衡、有噪声的标签、训练集和测试集分布不均衡；
```



## 解决欠拟合

```
（1）让模型更大：给模型加入更多的层 eg.ResNet-50 -> resNet-101，每层中更多的单元；

（2）减少正则化

（3）错误分析：（训练集和测试集的分布偏差）测试时候出现问题进行分析，训练集缺少哪些情况导致错误，后续将在训练集中加入此类数据纠正偏差；

（4）改进模型架构

（5）调节超参数：手动优化、网格搜索、随机搜索、由粗到细、贝叶斯优化；

（6）加入更多特征
```



## 解决过拟合

```
增加训练集样本，正则化
```



## 工程调参

```
3x3 conv是CNN主流组件（3x3Conv有利于保持图像性质）；

卷积核权重初始化使用xavier（Tanh）或者He normal（ReLU，PyTorch默认） ；

Batch Normalization或者Group Normalization；

使用ACNet的卷积方式；

cv2要比os读取图片速度快

加速训练pin_memory=true,work_numbers=4(卡的数量x4)，data.to(device,  no_blocking=True)

学习率和动量：使用较大的学习率+大的动量可以加快模型的训练且快速收敛

Adam learning rate： 3e-4

L2正则化：L2千万不要调太大，不然特别难训练；L2也不能太小，不然过拟合严重；即使正确地使用正则化强度，也会导致验证集前期不稳定甚至呈现训练不到的现象，但是之后就会稳定下来

优化器+学习率策略+momentum：

	1.SGD+momentum在大学习率+大动量的时候效果更好

	2.不管是SGD还是Adam还是AdamW，学习率的调整都对他们有帮助

	3.带有momentum的SGD加余弦退火收敛更快且更加稳定

	4.学习率最好设定好下限，不然后期会训练不动


把数据放内存里，降低 io 延迟

sudo mount tmpfs /path/to/your/data -t tmpfs -o size=30G


动态查看GPU利用率

watch -n 1 nvidia-smi


在显存大小固定情况下num_workers和batchsize是反比例关系
```



## 迁移学习

```
- 如果训练集小，训练数据与预训练数据相似，那么我们可以冻住卷积层，直接训练全连接层；
- 如果训练集小，训练数据与预训练数据不相似，那么必须从头训练卷积层及全连接层；
- 如果训练集大，训练数据与预训练数据相似，那么我们可以使用预训练的权重参数初始化网络，然后从头开始训练；
- 如果训练集大，训练数据与预训练数据不相似，那么我们可以使用预训练的权重参数初始化网络，然后从头开始训练或者完全不使用预训练权重，重新开始从头训练（推荐）；

值得注意的是，对于大数据集，不推荐冻住卷积层，直接训练全连接层的方式，这可能会对性能造成很大影响；
```



## Anaconda


```
# 清理缓存
conda clean -a

# 安装requirements里面的版本
conda install --yes --file requirements.txt

# 测试cuda是否可用
import torch
import torchvision
print(torch.cuda.is_available())
print(torch.version.cuda)

# 删除conda环境
conda remove -n name --all

# conda换源记得去掉default，添加pytorch
```



```
# conda 创建环境 + 装cuda + PyTorch

conda create -n name python=3.8
conda install cudatoolkit=10.1
conda install cudnn
使用pytorch官网的pip/conda命令装torch和torchvision

```





## Batch Normalization 改进

BN（Batch Normalization）几乎是目前神经网络的必选组件，但是使用BN有两个前提要求：

1. minibatch和全部数据同分布。因为训练过程每个minibatch从整体数据中均匀采样，不同分布的话minibatch的均值和方差和训练样本整体的均值和方差是会存在较大差异的，在测试的时候会严重影响精度。
2. batchsize不能太小，否则效果会较差，论文给的一般性下限是32。

BN有两个优点：

- 降低对初始化、学习率等超参的敏感程度，因为每层的输入被BN拉成相对稳定的分布，也能加速收敛过程。
- 应对梯度饱和和梯度弥散，主要是对于使用sigmoid和tanh的激活函数的网络。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfrkQgT3nibY56wze3Rx5w17KibyCvBLicZZ30icnGByAuiavhtFxBtDXwGV3uibia5rsJCRfPIPSmwdYypFA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**batchsize过小的场景**

实际的项目中，经常遇到需要处理的图片尺度过大的场景，例如我们使用500w像素甚至2000w像素的工业相机进行数据采集，500w的相机采集的图片尺度就是2500X2000左右。而对于微小的缺陷检测、高精度的关键点检测或小物体的目标检测等任务，我们一般不太想粗暴降低输入图片的分辨率，这样违背了我们使用高分辨率相机的初衷，也可能导致丢失有用特征。在算力有限的情况下，我们的batchsize就无法设置太大，甚至只能为1或2。小的batchsize会带来很多训练上的问题，其中BN问题就是最突出的。虽然大batchsize训练是一个共识，但是现实中可能无法具有充足的资源，因此我们需要一些处理手段。



首先Batch Normalization 中的Normalization被称为标准化，通过将数据进行平和缩放拉到一个特定的分布。BN就是在batch维度上进行数据的标准化。BN的引入是用来解决 internal covariate shift 问题，即训练迭代中网络激活的分布的变化对网络训练带来的破坏。BN通过在每次训练迭代的时候，利用minibatch计算出的当前batch的均值和方差，进行标准化来缓解这个问题。



**两种解决方式：BRN + CBN**

**BRN**

本文的核心思想就是：训练过程中，由于batchsize较小，当前minibatch统计到的均值和方差与全部数据有差异，那么就对当前的均值和方差进行修正。修正的方法主要是利用到通过滑动平均收集到的全局均值和标准差。

**CBN**

本文认为BRN的问题在于它使用的全局均值和标准差不是当前网络权重下获取的，因此不是exactly正确的，所以batchsize再小一点，例如为1或2时就不太work了。本文使用泰勒多项式逼近原理来修正当前的均值和标准差，同样也是间接利用了全局的均值和方差信息。简述就是：当前batch的均值和方差来自之前的K次迭代均值和方差的平均，由于网络权重一直在更新，所以不能直接粗暴求平均。本文而是利用泰勒公式估计前面的迭代在当前权重下的数值。



## Pytorch提速



**1.找到训练过程的瓶颈**

```
https://pytorch.org/docs/stable/bottleneck.html
```



**2.图片解码**

PyTorch中默认使用的是Pillow进行图像的解码，但是其效率要比Opencv差一些，如果图片全部是JPEG格式，可以考虑使用TurboJpeg库解码。具体速度对比如下图所示：

![图片](https://mmbiz.qpic.cn/mmbiz_png/SdQCib1UzF3szsSYrT2hU8JJhwlWibS4D4VHHTZKQXPuWDzfOiaaN26v6egU70QOWv5p4yUonYPPMqBnyXiaYlhqZg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)





**3.数据增强加速**

在PyTorch中，通常使用transformer做图片分类任务的数据增强，而其调用的是CPU做一些Crop、Flip、Jitter等操作。如果你通过观察发现你的CPU利用率非常高，GPU利用率比较低，那说明瓶颈在于CPU预处理，可以使用Nvidia提供的DALI库在GPU端完成这部分数据增强操作。

```
https://github.com/NVIDIA/DALI

Dali文档：https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/index.html
```



**4.data Prefetch**

```
https://zhuanlan.zhihu.com/p/66145913
https://zhuanlan.zhihu.com/p/97190313
```



**5.learning rate schedule**

**Cyclical Learning Rates** and the **1Cycle learning rate schedule** are both methods introduced by Leslie N. Smith. Essentially, the 1Cycle learning rate schedule looks something like this:

![img](https://efficientdl.com/content/images/2020/11/art5_lr_schedule.png)

Sylvain writes: 

> 1cycle consists of  two steps of equal lengths, one going from a lower learning rate to a higher one than go back to the minimum. The maximum should be the value picked with the Learning Rate Finder, and the lower one can be ten times lower. Then, the length of this cycle should be slightly less than the total number of epochs, and, in the last part of training, we should allow the learning rate to decrease more than the minimum, by several orders of magnitude.



**PyTorch implements** both of these methods `torch.optim.lr_scheduler.CyclicLR` and `torch.optim.lr_scheduler.OneCycleLR` see [the documentation](https://pytorch.org/docs/stable/optim.html).

**One drawback** of these schedulers is that they introduce a number of additional hyperparameters.

**Why does this work** One[ possible explanation](https://arxiv.org/pdf/1506.01186.pdf)might be that regularly increasing the learning rate helps to traverse [saddle points in the loss landscape ](https://papers.nips.cc/paper/2015/file/430c3626b879b4005d41b8a46172e0c0-Paper.pdf)more quickly.



**6.Use multiple workers and pinned memory in `DataLoader`**

When using [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), set `num_workers > 0`, rather than the default value of 0, and `pin_memory=True`, rather than the default value of `False`. 

A rule of thumb that [people are using ](https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5)to choose **the number of workers is to set it to four times the number of available GPUs** with both **a larger and smaller number of workers leading to a slow down.**



**7.Max out the batch size**

It seems like using the largest batch size your GPU memory permits **will accelerate your training** . Note that you will also have to adjust other hyperparameters, such as the learning rate, if you modify the batch size. **A rule of thumb here is to double the learning rate as you double the batch size.**

 **Might lead to solutions that generalize worse than those trained with smaller batches.**



**8. Use Automatic Mixed Precision (AMP)**

The release of PyTorch 1.6 included a native implementation of Automatic Mixed Precision training to PyTorch. The main idea here is that certain operations can be run faster and without a loss of accuracy at semi-precision (FP16) rather than in the single-precision (FP32) used elsewhere. AMP, then, automatically decide which operation should be executed in which format. This allows both for faster training and a smaller memory footprint.



**9.Using another optimizer**

AdamW is Adam with weight decay (rather than L2-regularization) and is now available natively in PyTorch as 
`torch.optim.AdamW`. AdamW seems to consistently outperform Adam in terms of both the error achieved and the training time. 

Both Adam and AdamW work well with the 1Cycle policy described above.



**10.Turn on cudNN benchmarking**

If your model architecture remains fixed and your input size stays constant, setting `torch.backends.cudnn.benchmark = True` might be beneficial. 



**11.Beware of frequently transferring data between CPUs and GPUs**

Beware of frequently transferring tensors from a GPU to a CPU using`tensor.cpu()` and vice versa using `tensor.cuda()` as these are relatively expensive. The same applies for `.item()` and `.numpy()` – use `.detach()` instead.

If you are creating a new tensor, you can also directly assign it to your GPU using the keyword argument `device=torch.device('cuda:0')`.

If you do need to transfer data, using `.to(device, non_blocking=True)`, might be useful [as long as you don't have any synchronization points](https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4) after the transfer.



**12.Use gradient/activation checkpointing**

> Checkpointing works by trading compute for memory. Rather than storing all intermediate activations of the entire computation graph for computing backward, **the checkpointed part does not save intermediate activations, and instead recomputes them in backward pass.** It can be applied on any part of a model.

> Specifically, in the forward pass, `function` will run in [`torch.no_grad()`](https://pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad)manner, i.e., not storing the intermediate activations. Instead, the forward pass saves the inputs tuple and the `function` parameter. In the backwards pass, the saved inputs and `function` is retrieved, and the forward pass is computed on `function` again, now tracking the intermediate activations, and then the gradients are calculated using these activation values.

So while this will might slightly increase your run time for a given batch size, you'll significantly reduce your memory footprint. This in turn will allow you to further increase the batch size you're using allowing for better GPU utilization.

While checkpointing is implemented natively as `torch.utils.checkpoint`([docs](https://pytorch.org/docs/stable/checkpoint.html)), it does seem to take some thought and effort to implement properly. 



**13.Use gradient accumulation**

Another approach to increasing the batch size is to accumulate gradients across multiple `.backward()` passes before calling `optimizer.step()`.

This method was developed mainly to circumvent GPU memory limitations and I'm not entirely clear on the trade-off between having additional `.backward()` loops.



**14.Use Distributed Data Parallel for multi-GPU training**

one simple one is to use `torch.nn.DistributedDataParallel` rather than `torch.nn.DataParallel`. By doing so, each GPU will be driven by a dedicated CPU core avoiding the GIL issues of `DataParallel`.

https://pytorch.org/tutorials/beginner/dist_overview.html



**15.Set gradients to None rather than 0**

Use `.zero_grad(set_to_none=True)` rather than `.zero_grad()`.

Doing so will let the memory allocator handle the gradients rather than actively setting them to 0. This will lead to yield a *modest* speed-up as they say in the [documentation](https://pytorch.org/docs/stable/optim.html), so don't expect any miracles.

Watch out, **doing this is not side-effect free**! Check the docs for the details on this.



**16.Use `.as_tensor()` rather than `.tensor()`**

`torch.tensor()` always copies data. If you have a numpy array that you want to convert, use `torch.as_tensor()` or `torch.from_numpy()` to avoid copying the data.



**17.Use gradient clipping**









## 2020珠港澳人工智能算法大赛



**数据集**

```
1.图像尺寸不一、近景和远景目标尺度差异大：
		图片尺寸不一，相差较大。一方面，由于计算资源和算法性能的限制，大尺寸的图像不能作为网络的输入，而单纯将原图像缩放到小图会使得目标丢失大量信息。另一方面，图像中近景和远景的目标尺度差异大，对于检测器来说，是个巨大的挑战。

2.目标在图像中分布密集，并且遮挡严重：
		数据集均是利用摄像头从真实场景采集，部分数据的目标密集度较大。都出现了频繁出现遮挡现象，目标的漏检情况相对严重。
```



```
anchor-based：🔥
1）优点：加入了先验知识，模型训练相对稳定；密集的anchor box可有效提高召回率，对于小目标检测来说提升非常明显。
2）缺点：对于多类别目标检测，超参数scale和aspect ratio相对难设计；冗余box非常多，可能会造成正负样本失衡；在进行目标类别分类时，超参IOU阈值需根据任务情况调整。

anchor-free：
1）优点：计算量减少；可灵活使用。
2）缺点：存在正负样本严重不平衡；两个目标中心重叠的情况下，造成语义模糊性；检测结果相对不稳定。
```

考虑到项目情况：

1）属于小类别检测，目标的scale和aspect ratio都在一定范围之内，属可控因素。

2）比赛数据中存在很多目标遮挡情况，这有可能会造成目标中心重新，如果采用anchor-free，会造成语义模糊性；

3）scale和aspect ratio可控，那么超参IOU调整相对简单；

4）对模型部署没有特殊要求，因此，部署方案相对较多，模型性能有很大改进。



**项目分析**

```
首先根据训练数据集进行分析，在10537张训练图像中，总共有12个组合类别、15个场景、18304个目标框。存在以下三种情况：
（1）样本不平衡，12个组合中，仅长袖-长裤组合占总数据的76.45%；
（2）场景样本不均衡，商场、工厂和街拍等五个场景中占比86.18%；
（3）多种状态行人，例如重叠、残缺、和占比小且遮挡。


另外，要权衡检测分类的精度和模型运行的速度，因此我们决定选用检测分类精度较好的目标检测框架，同时使用模型压缩和模型加速方法完成加速。其主体思路为：
（1） 目标检测框架：基于YOLOv5的one-stage检测框架；
（2） 模型压缩：基于BN放缩因子修剪主干网络；Slimming利用通道稀疏化的方法可以达到1）减少模型大小；2）减少运行时内存占用；3）在不影响精度的同时，降低计算操作数。
（3） 模型加速：TensorRT封装部署。在确保精度相对不变的情况下，采用FP16比FP32速度可提升1.5倍左右。另外，TensorRT是一个高性能的深度学习推理优化器，可以为深度学习应用提供低延迟、高吞吐的部署推理。

使用albumentations完成数据增强（mosaic数据增强会出现正样本数据被破坏情况）



模型预测存在大量的误检和漏检。这些漏检和无意义的检测结果大幅降低了模型的性能。我们将上述问题归纳为以下两个方面的原因：
1、YOLOv5s无论是网络宽度和网络深度都较小，学习能力相对较弱。小摊位占道和其他正常车辆十分相似，容易对分类器造成混淆，从而产生误检；
2、训练和测试时输入模型的图像尺度不合适。图像经过缩放后，目标的尺度也随之变小，导致远景中人的小摊贩等区域被大量遗漏；

首先，从图像预处理方面，使用随机中心裁剪方式切图进行训练。随机窗口切图是一种常用的大图像处理方式，这样可以有效地保留图像的高分辨率信息，不同大小的目标，另一方面采用多尺度训练，这样使得网络获得的信息更加丰富。如果某个目标处于切图边界，根据目标框的与图片的大小比例来决定是否保留。另外，我们还采用了随机几何变换、颜色扰动、翻转、多尺度、mixup、GridMask、Mosaic等数据增广方式，都可提高模型的泛化能力和小目标检测率。

其次，从优化器层面来讲，我们尝试了优化器梯度归一化和SAM优化器。
优化器梯度归一化有三个好处：（1）加速收敛；（2）防止梯度爆炸；（3）防止过拟合；
SAM优化器具有固有的鲁棒性。


3255张测试集中1080*1920尺寸的图像与其他尺寸的图像比例约为7:3。于是我们TensorRT部署时，模型使用输入大小为384*640比640*640检测率更优。因为1080*1920直接resize为640*640，一方面会到值目标变形，另一面，目标变得更小。
```



**结论**

```
1、 数据分析对于训练模型至关重要。数据不平衡、图像尺寸和目标大小不一、目标密集和遮挡等问题，应选用对应的baseline和应对策略。例如，数据不平衡可尝试过采样、focal loss、数据增强等策略；图像尺寸和目标大小不一可采用多尺度、数据裁剪等方法。
2、 针对算法精度和性能两者取舍来说，可先实验网络大小和输入图片大小对模型结果的影响，不同任务和不同数据情况，两者相差较大。所以不能一味为了提高速度，单纯压缩网络大小；
3、 针对性能要求时，可采用TensorRT等方式部署模型，也可采用模型压缩等方式，这样可在保证速度的前提下，使用较大网络，提升模型精度。
```

