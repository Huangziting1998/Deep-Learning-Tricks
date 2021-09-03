



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



- **找到训练过程的瓶颈**

```
https://pytorch.org/docs/stable/bottleneck.html
```



- **图片解码**

PyTorch中默认使用的是Pillow进行图像的解码，但是其效率要比Opencv差一些，如果图片全部是JPEG格式，可以考虑使用TurboJpeg库解码。具体速度对比如下图所示：

![图片](https://mmbiz.qpic.cn/mmbiz_png/SdQCib1UzF3szsSYrT2hU8JJhwlWibS4D4VHHTZKQXPuWDzfOiaaN26v6egU70QOWv5p4yUonYPPMqBnyXiaYlhqZg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)





- **数据增强加速**

在PyTorch中，通常使用transformer做图片分类任务的数据增强，而其调用的是CPU做一些Crop、Flip、Jitter等操作。如果你通过观察发现你的**CPU利用率非常高，GPU利用率比较低，**那说明瓶颈在于CPU预处理，可以使用Nvidia提供的DALI库在GPU端完成这部分数据增强操作。

```
https://github.com/NVIDIA/DALI

Dali文档：https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/index.html
```



- **data Prefetch & Use multiple workers and pinned memory in `DataLoader`**

When using [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), set `num_workers > 0`, rather than the default value of 0, and `pin_memory=True`, rather than the default value of `False`. 

A rule of thumb that [people are using ](https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5)to choose **the number of workers is to set it to four times the number of available GPUs** with both **a larger and smaller number of workers leading to a slow down.**

```python
 DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                              pin_memory=True, prefetch_factor=2)  # prefetch works when pin_memory > 0
```



- **learning rate schedule**

**Cyclical Learning Rates** and the **1Cycle learning rate schedule** are both methods introduced by Leslie N. Smith. Essentially, the 1Cycle learning rate schedule looks something like this:

![img](https://efficientdl.com/content/images/2020/11/art5_lr_schedule.png)

Sylvain writes: 

> 1cycle consists of  two steps of equal lengths, one going from a lower learning rate to a higher one than go back to the minimum. The maximum should be the value picked with the Learning Rate Finder, and the lower one can be ten times lower. Then, the length of this cycle should be slightly less than the total number of epochs, and, in the last part of training, we should allow the learning rate to decrease more than the minimum, by several orders of magnitude.



**PyTorch implements** both of these methods `torch.optim.lr_scheduler.CyclicLR` and `torch.optim.lr_scheduler.OneCycleLR` see [the documentation](https://pytorch.org/docs/stable/optim.html).

**One drawback** of these schedulers is that they introduce a number of additional hyperparameters.

**Why does this work** One[ possible explanation](https://arxiv.org/pdf/1506.01186.pdf)might be that regularly increasing the learning rate helps to traverse [saddle points in the loss landscape ](https://papers.nips.cc/paper/2015/file/430c3626b879b4005d41b8a46172e0c0-Paper.pdf)more quickly.



- **Max out the batch size**

It seems like using the largest batch size your GPU memory permits **will accelerate your training** . Note that you will also have to adjust other hyperparameters, such as the learning rate, if you modify the batch size. **A rule of thumb here is to double the learning rate as you double the batch size.**

 **Might lead to solutions that generalize worse than those trained with smaller batches.**



- **Use Automatic Mixed Precision (AMP)**

The release of PyTorch 1.6 included a native implementation of Automatic Mixed Precision training to PyTorch. The main idea here is that certain operations can be run faster and without a loss of accuracy at semi-precision (FP16) rather than in the single-precision (FP32) used elsewhere. AMP, then, automatically decide which operation should be executed in which format. This allows both for faster training and a smaller memory footprint.



- **Using another optimizer**

AdamW is Adam with weight decay (rather than L2-regularization) and is now available natively in PyTorch as 
`torch.optim.AdamW`. AdamW seems to consistently outperform Adam in terms of both the error achieved and the training time. 

Both Adam and AdamW work well with the 1Cycle policy described above.



- **Turn on cudNN benchmarking**

If your model architecture remains fixed and your input size stays constant, setting `torch.backends.cudnn.benchmark = True` might be beneficial. 



- **Beware of frequently transferring data between CPUs and GPUs**

Beware of frequently transferring tensors from a GPU to a CPU using`tensor.cpu()` and vice versa using `tensor.cuda()` as these are relatively expensive. The same applies for `.item()` and `.numpy()` – use `.detach()` instead.

If you are creating a new tensor, you can also directly assign it to your GPU using the keyword argument `device=torch.device('cuda:0')`.

If you do need to transfer data, using `.to(device, non_blocking=True)`, might be useful [as long as you don't have any synchronization points](https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4) after the transfer.



- **Use gradient/activation checkpointing**

> Checkpointing works by trading compute for memory. Rather than storing all intermediate activations of the entire computation graph for computing backward, **the checkpointed part does not save intermediate activations, and instead recomputes them in backward pass.** It can be applied on any part of a model.

> Specifically, in the forward pass, `function` will run in [`torch.no_grad()`](https://pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad)manner, i.e., not storing the intermediate activations. Instead, the forward pass saves the inputs tuple and the `function` parameter. In the backwards pass, the saved inputs and `function` is retrieved, and the forward pass is computed on `function` again, now tracking the intermediate activations, and then the gradients are calculated using these activation values.

So while this will might slightly increase your run time for a given batch size, you'll significantly reduce your memory footprint. This in turn will allow you to further increase the batch size you're using allowing for better GPU utilization.

While checkpointing is implemented natively as `torch.utils.checkpoint`([docs](https://pytorch.org/docs/stable/checkpoint.html)), it does seem to take some thought and effort to implement properly. 



- **Use gradient accumulation**

Another approach to increasing the batch size is to accumulate gradients across multiple `.backward()` passes before calling `optimizer.step()`.

This method was developed mainly to circumvent GPU memory limitations and I'm not entirely clear on the trade-off between having additional `.backward()` loops.



- **Use Distributed Data Parallel for multi-GPU training**

one simple one is to use `torch.nn.DistributedDataParallel` rather than `torch.nn.DataParallel`. By doing so, each GPU will be driven by a dedicated CPU core avoiding the GIL issues of `DataParallel`.

https://pytorch.org/tutorials/beginner/dist_overview.html



- **Set gradients to None rather than 0**

Use `.zero_grad(set_to_none=True)` rather than `.zero_grad()`.

Doing so will let the memory allocator handle the gradients rather than actively setting them to 0. This will lead to yield a *modest* speed-up as they say in the [documentation](https://pytorch.org/docs/stable/optim.html), so don't expect any miracles.

Watch out, **doing this is not side-effect free**! Check the docs for the details on this.



- **Use `.as_tensor()` rather than `.tensor()`**

`torch.tensor()` always copies data. If you have a numpy array that you want to convert, use `torch.as_tensor()` or `torch.from_numpy()` to avoid copying the data.



- **Use gradient clipping**

In PyTorch this can be done using `torch.nn.utils.clip_grad_norm_`([documentation](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html#torch.nn.utils.clip_grad_norm_)).



- **Turn off bias before BatchNorm**

This is a very simple one: turn off the bias of layers before BatchNormalization layers. For a 2-D convolutional layer, this can be done by setting the bias keyword to False: `torch.nn.Conv2d(..., bias=False, ...)`.



- **Turn off gradient computation during validation**

This one is straightforward: set `torch.no_grad()` during validation.



- **Use input and batch normalization**

You're probably already doing this but you might want to double-check:

- Are you [normalizing](https://pytorch.org/docs/stable/torchvision/transforms.html) your input? 
- Are you using [batch-normalization](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)?

And [here's](https://stats.stackexchange.com/questions/437840/in-machine-learning-how-does-normalization-help-in-convergence-of-gradient-desc) a reminder of why you probably should.



## Paper写作



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



