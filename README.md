



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



## Pytorch提速



**找到训练过程的瓶颈**

```
https://pytorch.org/docs/stable/bottleneck.html
```



**图片解码**

PyTorch中默认使用的是Pillow进行图像的解码，但是其效率要比Opencv差一些，如果图片全部是JPEG格式，可以考虑使用TurboJpeg库解码。具体速度对比如下图所示：

![图片](https://mmbiz.qpic.cn/mmbiz_png/SdQCib1UzF3szsSYrT2hU8JJhwlWibS4D4VHHTZKQXPuWDzfOiaaN26v6egU70QOWv5p4yUonYPPMqBnyXiaYlhqZg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)





**数据增强加速**

在PyTorch中，通常使用transformer做图片分类任务的数据增强，而其调用的是CPU做一些Crop、Flip、Jitter等操作。如果你通过观察发现你的CPU利用率非常高，GPU利用率比较低，那说明瓶颈在于CPU预处理，可以使用Nvidia提供的DALI库在GPU端完成这部分数据增强操作。

```
https://github.com/NVIDIA/DALI

Dali文档：https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/index.html
```



**data Prefetch**

```
https://zhuanlan.zhihu.com/p/66145913
https://zhuanlan.zhihu.com/p/97190313
```



**learning rate schedule**

**Cyclical Learning Rates** and the **1Cycle learning rate schedule** are both methods introduced by Leslie N. Smith. Essentially, the 1Cycle learning rate schedule looks something like this:

![img](https://efficientdl.com/content/images/2020/11/art5_lr_schedule.png)

Sylvain writes: 

> 1cycle consists of  two steps of equal lengths, one going from a lower learning rate to a higher one than go back to the minimum. The maximum should be the value picked with the Learning Rate Finder, and the lower one can be ten times lower. Then, the length of this cycle should be slightly less than the total number of epochs, and, in the last part of training, we should allow the learning rate to decrease more than the minimum, by several orders of magnitude.



**PyTorch implements** both of these methods `torch.optim.lr_scheduler.CyclicLR` and `torch.optim.lr_scheduler.OneCycleLR` see [the documentation](https://pytorch.org/docs/stable/optim.html).

**One drawback** of these schedulers is that they introduce a number of additional hyperparameters.

**Why does this work** One[ possible explanation](https://arxiv.org/pdf/1506.01186.pdf)might be that regularly increasing the learning rate helps to traverse [saddle points in the loss landscape ](https://papers.nips.cc/paper/2015/file/430c3626b879b4005d41b8a46172e0c0-Paper.pdf)more quickly.



**Use multiple workers and pinned memory in `DataLoader`**

When using [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), set `num_workers > 0`, rather than the default value of 0, and `pin_memory=True`, rather than the default value of `False`. 

A rule of thumb that [people are using ](https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5)to choose **the number of workers is to set it to four times the number of available GPUs** with both **a larger and smaller number of workers leading to a slow down.**



**Max out the batch size**

It seems like using the largest batch size your GPU memory permits **will accelerate your training** . Note that you will also have to adjust other hyperparameters, such as the learning rate, if you modify the batch size. **A rule of thumb here is to double the learning rate as you double the batch size.**

 **Might lead to solutions that generalize worse than those trained with smaller batches.**



**Use Automatic Mixed Precision (AMP)**

The release of PyTorch 1.6 included a native implementation of Automatic Mixed Precision training to PyTorch. The main idea here is that certain operations can be run faster and without a loss of accuracy at semi-precision (FP16) rather than in the single-precision (FP32) used elsewhere. AMP, then, automatically decide which operation should be executed in which format. This allows both for faster training and a smaller memory footprint.



**Using another optimizer**

AdamW is Adam with weight decay (rather than L2-regularization) and is now available natively in PyTorch as 
`torch.optim.AdamW`. AdamW seems to consistently outperform Adam in terms of both the error achieved and the training time. 

Both Adam and AdamW work well with the 1Cycle policy described above.



**Turn on cudNN benchmarking**

If your model architecture remains fixed and your input size stays constant, setting `torch.backends.cudnn.benchmark = True` might be beneficial. 



**Beware of frequently transferring data between CPUs and GPUs**

Beware of frequently transferring tensors from a GPU to a CPU using`tensor.cpu()` and vice versa using `tensor.cuda()` as these are relatively expensive. The same applies for `.item()` and `.numpy()` – use `.detach()` instead.

If you are creating a new tensor, you can also directly assign it to your GPU using the keyword argument `device=torch.device('cuda:0')`.

If you do need to transfer data, using `.to(device, non_blocking=True)`, might be useful [as long as you don't have any synchronization points](https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4) after the transfer.



**Use gradient/activation checkpointing**

> Checkpointing works by trading compute for memory. Rather than storing all intermediate activations of the entire computation graph for computing backward, **the checkpointed part does not save intermediate activations, and instead recomputes them in backward pass.** It can be applied on any part of a model.

> Specifically, in the forward pass, `function` will run in [`torch.no_grad()`](https://pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad)manner, i.e., not storing the intermediate activations. Instead, the forward pass saves the inputs tuple and the `function` parameter. In the backwards pass, the saved inputs and `function` is retrieved, and the forward pass is computed on `function` again, now tracking the intermediate activations, and then the gradients are calculated using these activation values.

So while this will might slightly increase your run time for a given batch size, you'll significantly reduce your memory footprint. This in turn will allow you to further increase the batch size you're using allowing for better GPU utilization.

While checkpointing is implemented natively as `torch.utils.checkpoint`([docs](https://pytorch.org/docs/stable/checkpoint.html)), it does seem to take some thought and effort to implement properly. 



**Use gradient accumulation**

Another approach to increasing the batch size is to accumulate gradients across multiple `.backward()` passes before calling `optimizer.step()`.

This method was developed mainly to circumvent GPU memory limitations and I'm not entirely clear on the trade-off between having additional `.backward()` loops.



**Use Distributed Data Parallel for multi-GPU training**

one simple one is to use `torch.nn.DistributedDataParallel` rather than `torch.nn.DataParallel`. By doing so, each GPU will be driven by a dedicated CPU core avoiding the GIL issues of `DataParallel`.

https://pytorch.org/tutorials/beginner/dist_overview.html



**Set gradients to None rather than 0**

Use `.zero_grad(set_to_none=True)` rather than `.zero_grad()`.

Doing so will let the memory allocator handle the gradients rather than actively setting them to 0. This will lead to yield a *modest* speed-up as they say in the [documentation](https://pytorch.org/docs/stable/optim.html), so don't expect any miracles.

Watch out, **doing this is not side-effect free**! Check the docs for the details on this.



**Use `.as_tensor()` rather than `.tensor()`**

`torch.tensor()` always copies data. If you have a numpy array that you want to convert, use `torch.as_tensor()` or `torch.from_numpy()` to avoid copying the data.



**Use gradient clipping**

In PyTorch this can be done using `torch.nn.utils.clip_grad_norm_`([documentation](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html#torch.nn.utils.clip_grad_norm_)).



**urn off bias before BatchNorm**

This is a very simple one: turn off the bias of layers before BatchNormalization layers. For a 2-D convolutional layer, this can be done by setting the bias keyword to False: `torch.nn.Conv2d(..., bias=False, ...)`.



**Turn off gradient computation during validation**

This one is straightforward: set `torch.no_grad()` during validation.



**Use input and batch normalization**

You're probably already doing this but you might want to double-check:

- Are you [normalizing](https://pytorch.org/docs/stable/torchvision/transforms.html) your input? 
- Are you using [batch-normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)?

And [here's](https://stats.stackexchange.com/questions/437840/in-machine-learning-how-does-normalization-help-in-convergence-of-gradient-desc) a reminder of why you probably should.



## Paper写作



```
Google Doc检查写作
```



**Introduction**

```
- Problem definition
- Previous methods and their limits
- 简单描述你是提出了什么技术来 overcome 上面的 limits
- 一个图，非常 high-level 的解释前人工作的 limits 和你的工作怎么解决了这些 limits，最好让人30秒内完全看懂
- 最后一段如今大都是：In summary, this paper makes three contributions: First work to 解决什么 limits；提出了什么 novel 的技术；outperform 了 state-of-the-art 多少
```



**Related Work**

```
一般三五个 subsection，分别 review 下相关的 topics，同样不光讲 previous work 做了什么，更要讲自己的方法跟前人工作有啥不同
```



**Method**

```
- 这是文章的主体，按照你觉得最容易让别人看懂的方式来讲
- 可以第一个 subsection 是 overview，formulate 一下你的 problem 给出 notation，配一个整体 framework 的图，图里面的字体不能太大或者太小看不清，要有些细节，让人光看图就能明白你的方法是怎么回事，但不要过于复杂，让人在不超过 2 分钟的时间看完这张图
- 然后几个 subsection 具体介绍你的方法或者模型；如果 testing 跟 training 不太一样，最后一个 subsection 介绍 inference 时候的不同，通常是一些 post-processing 操作
```



**Experiment**

```
- Datasets
- Implementation details such as pre-processing process, training recipe
- Evaluation metrics
- Comparisons with state-of-the-art
- Detailed analysis
- Alternative design choice exploration
- Ablation studies
- Visualization examples
```



**Conclusion (and Future Work)**



**Abstract**

```
是全文的精简，建议在 paper 写完第一稿差不多成型了，有定下来的成熟的 storyline 了，再去写 abstract；大概就是用一两句话分别概括 paper 里面每个 section，然后串起来
```





## 2020珠港澳人工智能算法大赛



**数据集**

```
1.图像尺寸不一、近景和远景目标尺度差异大：
图片尺寸不一，相差较大。一方面，由于计算资源和算法性能的限制，大尺寸的图像不能作为网络的输入，而单纯将原图像缩放到小图会使得目标丢失大量信息。另一方面，图像中近景和远景的目标尺度差异大，对于检测器来说，是个巨大的挑战。

2.目标在图像中分布密集，并且遮挡严重：
数据集均是利用摄像头从真实场景采集，部分数据的目标密集度较大。都出现了频繁出现遮挡现象，目标的漏检情况相对严重。
```



```
anchor-based：
1）优点：加入了先验知识，模型训练相对稳定；密集的anchor box可有效提高召回率，对于小目标检测来说提升非常明显。
2）缺点：对于多类别目标检测，超参数scale和aspect ratio相对难设计；冗余box非常多，可能会造成正负样本失衡；在进行目标类别分类时，超参IOU阈值需根据任务情况调整。

anchor-free：
1）优点：计算量减少；可灵活使用。
2）缺点：存在正负样本严重不平衡；两个目标中心重叠的情况下，造成语义模糊性；检测结果相对不稳定。
```



考虑到项目情况：

1. 属于小类别检测，目标的scale和aspect ratio都在一定范围之内，属可控因素;

2. 比赛数据中存在很多目标遮挡情况，这有可能会造成目标中心重新，如果采用anchor-free，会造成语义模糊性；

3. scale和aspect ratio可控，那么超参IOU调整相对简单；

4. 对模型部署没有特殊要求，因此，部署方案相对较多，模型性能有很大改进。



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





## ICPR 2020大规模商品图像识别挑战赛



人工智能零售系统需要快速地从图像和视频中自动识别出产品的存货单元(Stock Keeping Unit，SKU)级别的类别，然而，许多SKU级别的产品都是细粒度的，可以看出它们在视觉上是相似的。



**数据集**

```
JDAI构建了一个名为Products-10K的产品识别数据集，这是迄今为止最大的一个产品识别数据集，其中包含了约10000种经常被中国消费者购买的产品，涵盖了时尚、3C、食品、保健、家居用品等全品类。Products-10K提供了约150000张图片，10000个细粒度的SKU级别的标签，以及360个组别标签，经过数据分析可以总结该数据集有以下特点：

1) 大尺度，少样本
虽然提供了150000张图片，但是由于类别数比较多，大部分类别的图片数量都少于20张；

2) 类间距离小
大部分类别在视觉上比较相似；虽然看起来十分相似，但是却属于不同的SKU标签；

3) 类内距离大
同一个细粒度标签下的图片包含了商店场景和消费者场景，商店场景的背景比较简单，消费者场景背景比较复杂；（同一个SKU标签，大部分商店场景图片都是白色背景，但消费者拍摄的图片背景比较多样化）
```



**方案**

```
本次竞赛方案采用了resnest作为基础骨架网络进行特征提取，并且使用了GeM pooling对骨架网络最后一层特征进行池化，基于池化的向量进行group和SKU-level的多任务分类，分类器采用了CircleSoftmax调整类间间距，并且在每一个分类器之前引入了一个BNNeck的结构。Loss上采用了FocalLoss和CrossEntropy Loss联合训练的方式。

数据增强：采用了常规的翻转、随机擦除、颜色增强、AugMix等;

池化：Generalized Mean Pooling (GeM Pooling) ;

分类器：使用了全连接层构建基线模型，通过数据分析发现该数据集存在类内距离大，类间距离小等特点，因此借鉴了人脸识别常用的分类器CosFace和CircleSoftmax，通过在训练过程中引入调整分类超平面的方式，使得测试时的不同类别的特征更容易区分;

Loss设计：使用了Focal Loss和CrossEntropy Loss联合训练的方案，避免了Focal Loss需要调整超参和过度放大困难样本权重的问题;

模型融合：
```



## ACCV2020国际细粒度识别比赛



```
经过初步实验和对数据集的可视化，我们发现本次比赛主要存在有以下挑战：
- 55万的训练数据集中存在有大量的噪声数据
- 训练集中存在较多的图片标签错误
- 训练集与测试集不属于同一分布，且存在较大差异
- 训练集各类别图片数量呈长尾分布
- 细粒度挑战，类间差异小
```



**清洗噪声数据**

```
1）从1万张非三通道图片中人工挑出1000张左右的噪声图片 和 7000张左右正常图片，训练二分类噪声数据识别模型；           

2）使用1中的二分类模型预测全量50万训练数据，挑选出阈值大于0.9的噪声数据；

3）使用2中噪声数据迭代 1、2过程，完成噪声数据清洗。人工检查，清洗后的训练样本中噪声数据占比小于1%；
```



**清洗粗粒度标签错误数据**

本次竞赛5000类别中，仍有较多的属于两个不同细粒度的图片具有相同标签。如下图的人物合影、荒草都和青蛙属于同一标签。

```
1）交叉训练，将50万训练集拆成五分，每4分当训练集，一份当测试集，训练5个模型；
2）将训练好的模型分别对各自测试集进行预测，将测试集top5正确的数据归为正例，top5错误的数据归为反例；
3）收集正例数据，重新训练模型，对反例数据进行预测，反例数据中top5正确的数据拉回放入训练集； 
4）使用不同的优化方案、数据增强反复迭代步骤3直至稳定（没有新的正例数据产出）；
5）人工干预：从反例数据中拉回5%-10%左右的数据，人工check，挑选出正例数据放入训练集；
6）重复3、4步骤。
```



**清洗细粒度标签错误数据**

细粒度类别标签错误数据，图片与其他三张图片不属于同一类别，却具有相同标签。

**清洗方案：**

```
1）交叉训练，将清洗粗粒度错误标签后的训练集拆成五分，每4分当训练集，一份当测试集，训练5个模型。

2）将训练好的模型分别对各自测试集进行预测，将测试集top1正确的数据归为正例，top1错误的数据归为反例。

3）收集正例数据，重新训练模型，对反例数据进行预测，反例数据中top1正确的数据拉回放入训练集

4）使用不同的优化方案、数据增强反复迭代步骤3直至稳定（没有新的正例数据产出）。

5）人工干预：从反例数据中拉回5%-10%左右的数据，人工check，挑选出正例数据放入训练集

6）重复3、4步骤。
```



**清除低质量类别**

```
在数据集的5000个类别中，人工看了图片数量少于50的类别，剔除其中图片混乱，无法确认此类别的具体标签。
```



**数据增强**

训练集与测试集属于不同分布，为使模型能更好的泛化测试集，以及捕捉局部细节特征区分细粒度类别，我们采用如下数据增强组合：

```
- mixcut
- 随机颜色抖动
- 随机方向—镜像翻转；4方向随机旋转
- 随机质量—resize150~190，再放大到380；随机jpeg低质量有损压缩
- 随机crop
- 图片随机网格打乱重组
- 随机缩放贴图
```



**Backbones**

```
在模型选型上，只使用了如下backbones：EfficientNet-b4、EfficientNet-b5
```



**优化**

在模型优化方面，我们使用radam+sgd优化器，以及大的batch size训练（我们在实验中发现，使用大batch size比小batch size收敛更快，测试集精度更高） ，具体参数如下：

```
- label smooth 0.2
- base_lr=0.03
- radam+sgd
- cosine scheduler
- 分布式超大batch size（25*80=2000）训练
```



**知识蒸馏**

加上知识蒸馏，可以使我们的模型精度提升约1%：

```
- 50+w训练集加20w测试集 ，纯模型蒸馏，采用KLDivLoss 损失函数
- 50+w训练集，模型蒸馏（KLDivLoss）*0.5 +标签（CrossEntropyLoss）* 0.5
```



**Ensemble**

```
通过选取不同版本的数据集，以及以上不同的数据增强、数据均衡、蒸馏方法和模型结构，训练多个模型。
```





