# An overview of gradient descent optimization algorithms



https://ruder.io/optimizing-gradient-descent/index.html#otherrecentoptimizers



梯度下降法是最著名的优化算法之一，也是迄今优化神经网络时最常用的方法。在每一个最新的深度学习库中都包含了各种优化的梯度下降法的实现。然而，这些算法通常是作为黑盒优化器使用，因此，很难对其优点和缺点的进行实际的解释。

梯度下降法是最小化目标函数$J(θ)$的一种方法，其中，$θ∈ℝd$ 为模型参数，梯度下降法利用目标函数关于参数的梯度$∇θJ(θ)$的反方向更新参数。学习率$η$决定达到最小值或者局部最小值过程中所采用的步长的大小。即，我们沿着目标函数的斜面下降的方向，直到到达谷底。



## Gradient descent variants

梯度下降法有3中变形形式，它们之间的区别为我们在计算目标函数的梯度时使用到多少数据。根据数据量的不同，我们在参数更新的精度和更新过程中所需要的时间两个方面做出权衡。



### Batch gradient descent

Vanilla梯度下降法，又称为批梯度下降法（batch gradient descent），在整个训练数据集上计算损失函数关于参数$θ$的梯度：
$$
θ=θ−η⋅∇θJ(θ)
$$
因为在执行每次更新时，我们需要在整个数据集上计算所有的梯度，所以批梯度下降法的速度会很慢，同时，批梯度下降法无法处理超出内存容量限制的数据集。批梯度下降法同样也不能在线更新模型，即在运行的过程中，不能增加新的样本。

批梯度下降法的代码如下所示：

```
for i in range(nb_epochs):  
	params_grad = evaluate_gradient(loss_function, data, params)  
	params = params - learning_rate * params_grad
```

对于给定的epochs，首先，我们利用全部数据集计算损失函数关于参数向量`params`的梯度向量`params_grad`。注意，最新的深度学习库中提供了自动求导的功能，可以有效地计算关于参数梯度。如果你自己求梯度，那么，梯度检查是一个不错的主意。

然后，我们利用梯度的方向和学习率更新参数，学习率决定我们将以多大的步长更新参数。对于凸误差函数，批梯度下降法能够保证收敛到全局最小值，对于非凸函数，则收敛到一个局部最小值。



### Stochastic gradient descent

随机梯度下降法（stochastic gradient descent, SGD）根据每一条训练样本$x(i)$和标签$y(i)$更新参数：
$$
θ=θ−η⋅∇θJ(θ;x(i);y(i))
 
$$
对于大数据集，因为批梯度下降法在每一个参数更新之前，会对相似的样本计算梯度，所以在计算过程中会有冗余。而SGD在每一次更新中只执行一次，从而消除了冗余。因而，通常SGD的运行速度更快，同时，可以用于在线学习。SGD以高方差频繁地更新，导致目标函数出现如图1所示的剧烈波动。

![SGD fluctuation](https://ruder.io/content/images/2016/09/sgd_fluctuation.png)



与批梯度下降法的收敛会使得损失函数陷入局部最小相比，由于SGD的波动性，一方面，波动性使得SGD可以跳到新的和潜在更好的局部最优。另一方面，这使得最终收敛到特定最小值的过程变得复杂，因为SGD会一直持续波动。然而，已经证明当我们缓慢减小学习率，SGD与批梯度下降法具有相同的收敛行为，对于非凸优化和凸优化，可以分别收敛到局部最小值和全局最小值。与批梯度下降的代码相比，SGD的代码片段仅仅是在对训练样本的遍历和利用每一条样本计算梯度的过程中增加一层循环。在每一次循环中，我们shuffle训练样本。

```
for i in range(nb_epochs):
  np.random.shuffle(data)
  for example in data:
    params_grad = evaluate_gradient(loss_function, example, params)
    params = params - learning_rate * params_grad
```



### Mini-batch gradient descent

小批量梯度下降法最终结合了上述两种方法的优点，在每次更新时使用$n$个小批量训练样本：
$$
\theta = \theta - \eta \cdot \nabla_\theta J( \theta; x^{(i:i+n)}; y^{(i:i+n)})
$$
这种方法，a)减少参数更新的方差，这样可以得到更加稳定的收敛结果；b)可以利用最新的深度学习库中高度优化的矩阵优化方法，高效地求解每个小批量数据的梯度。通常，小批量数据的大小在50到256之间，也可以根据不同的应用有所变化。当训练神经网络模型时，小批量梯度下降法是典型的选择算法，当使用小批量梯度下降法时，也将其称为SGD。注意：在下文的改进的SGD中，为了简单，我们省略了参数$x^{(i:i+n)}; y^{(i:i+n)}$。

在代码中，不是在所有样本上做迭代，我们现在只是在大小为50的小批量数据上做迭代：

```
for i in range(nb_epochs):
  np.random.shuffle(data)
  for batch in get_batches(data, batch_size=50):
    params_grad = evaluate_gradient(loss_function, batch, params)
    params = params - learning_rate * params_grad
```



## Challenges

虽然Vanilla小批量梯度下降法并不能保证较好的收敛性，但是需要强调的是，这也给我们留下了如下的一些挑战：

- 选择一个合适的学习率可能是困难的。学习率太小会导致收敛的速度很慢，学习率太大会妨碍收敛，导致损失函数在最小值附近波动甚至偏离最小值；
- 学习率调整试图在训练的过程中通过例如退火的方法调整学习率，即根据预定义的策略或者当相邻两代之间的下降值小于某个阈值时减小学习率。然而，策略和阈值需要预先设定好，因此无法适应数据集的特点；
- 此外，对所有的参数更新使用同样的学习率。如果数据是稀疏的，同时，特征的频率差异很大时，我们也许不想以同样的学习率更新所有的参数，对于出现次数较少的特征，我们对其执行更大的学习率；
- 高度非凸误差函数普遍出现在神经网络中，在优化这类函数时，另一个关键的挑战是使函数避免陷入无数次优的局部最小值。Dauphin等指出出现这种困难实际上并不是来自局部最小值，而是来自鞍点，即那些在一个维度上是递增的，而在另一个维度上是递减的。这些鞍点通常被具有相同误差的点包围，因为在任意维度上的梯度都近似为0，所以SGD很难从这些鞍点中逃开。



## Gradient descent optimization algorithms

我们不会讨论在实际中不适合在高维数据集中计算的算法，例如[牛顿法](https://en.wikipedia.org/wiki/Newton's_method_in_optimization)的二阶方法。



### Momentum

SGD很难通过陡谷，即在一个维度上的表面弯曲程度远大于其他维度的区域，这种情况通常出现在局部最优点附近。在这种情况下，SGD摇摆地通过陡谷的斜坡，同时，沿着底部到局部最优点的路径上只是缓慢地前进：



![SGD without momentum](https://ruder.io/content/images/2015/12/without_momentum.gif)![SGD with momentum](https://ruder.io/content/images/2015/12/with_momentum.gif)



​						SGD without momentum																	SGD with momentum



动量法是一种帮助SGD在相关方向上加速并抑制摇摆的一种方法。动量法将历史步长的更新向量的一个分量$γ$增加到当前的更新向量中：
$$
\begin{align} 
\begin{split} 
v_t &= \gamma v_{t-1} + \eta \nabla_\theta J( \theta) \\ 
\theta &= \theta - v_t 
\end{split} 
\end{align}
$$
Notes：部分实现中交换了公式中的符号，动量项$γ$通常设置为0.9或者类似的值。

从本质上说，动量法，就像我们从山上推下一个球，球在滚下来的过程中累积动量，变得越来越快（直到达到终极速度，如果有空气阻力的存在，则$γ<1$）。同样的事情也发生在参数的更新过程中：对于在梯度点处具有相同的方向的维度，其动量项增大，对于在梯度点处改变方向的维度，其动量项减小。因此，我们可以得到更快的收敛速度，同时可以减少摇摆。



### Nesterov accelerated gradient

然而，球从山上滚下的时候，盲目地沿着斜率方向，往往并不能令人满意。我们希望有一个智能的球，这个球能够知道它将要去哪，以至于在重新遇到斜率上升时能够知道减速。

Nesterov加速梯度下降法（Nesterov accelerated gradient，NAG）是一种能够给动量项这样的预知能力的方法。我们知道，我们利用动量项$\gamma v_{t-1}$来更新参数$θ$。通过计算$\theta - \gamma v_{t-1}$能够告诉我们参数未来位置的一个近似值（梯度并不是完全更新），这也就是告诉我们参数大致将变为多少。通过计算关于参数未来的近似位置的梯度，而不是关于当前的参数$θ$的梯度，我们可以高效的求解 ：
$$
\begin{align} 
\begin{split} 
v_t &= \gamma v_{t-1} + \eta \nabla_\theta J( \theta - \gamma v_{t-1} ) \\ 
\theta &= \theta - v_t 
\end{split} 
\end{align}
$$
同时，我们设置动量项γ大约为0.9。动量法首先计算当前的梯度值（小的蓝色向量），然后在更新的累积梯度（大的蓝色向量）方向上前进一大步，Nesterov加速梯度下降法NAG首先在先前累积梯度（棕色的向量）方向上前进一大步，计算梯度值，然后做一个修正（红色的向量），从而完成NAG更新（绿色向量）。这个具有预见性的更新防止我们前进得太快，同时增强了算法的响应能力，这一点在很多的任务中对于RNN的性能提升有着重要的意义。

![SGD fluctuation](https://ruder.io/content/images/2016/09/nesterov_update_vector.png)



既然我们能够使得我们的更新适应误差函数的斜率以相应地加速SGD，我们同样也想要使得我们的更新能够适应每一个单独参数，以根据每个参数的重要性决定大的或者小的更新。



### Adagrad

Adagrad是这样的一种基于梯度的优化算法：让学习率适应参数，对于出现次数较少的特征，我们对其采用更大的学习率，对于出现次数较多的特征，我们对其采用较小的学习率。因此，Adagrad非常适合处理稀疏数据。Dean等人发现Adagrad能够极大提高了SGD的鲁棒性并将其应用于Google的大规模神经网络的训练，其中包含了[YouTube视频](http://www.wired.com/2012/06/google-x-neural-network/)中的猫的识别。此外，Pennington等人利用Adagrad训练Glove词向量，因为低频词比高频词需要更大的步长。

前面，我们每次更新所有的参数$θ$时，每一个参数$θ_i$都使用的是相同的学习率$η$。由于Adagrad在$t$时刻对每一个参数$θ_i$使用了不同的学习率，我们首先介绍Adagrad对每一个参数的更新，然后我们对其向量化。为了简洁，令$g_{t, i}$为在$t$时刻目标函数关于参数$θ_i$的梯度：
$$
g_{t, i} = \nabla_\theta J( \theta_{t, i} )
$$
在$t$时刻，对每个参数$θ_i$的更新过程变为：
$$
\theta_{t+1, i} = \theta_{t, i} - \eta \cdot g_{t, i}
$$
对于上述的更新规则，在$t$时刻，基于对$θ_i$计算过的历史梯度，Adagrad修正了对每一个参数$θ_i$的学习率：
$$
\theta_{t+1, i} = \theta_{t, i} - \dfrac{\eta}{\sqrt{G_{t, ii} + \epsilon}} \cdot g_{t, i}
$$
其中，$G_t∈ℝd×d$是一个对角矩阵，对角线上的元素$i,i$是直到$t$时刻为止，所有关于$θ_i$的梯度的平方和（Duchi等人将该矩阵作为包含所有先前梯度的外积的完整矩阵的替代，因为即使是对于中等数量的参数$d$，矩阵的均方根的计算都是不切实际的），$ϵ$是平滑项，用于防止除数为0（通常大约设置为1e−8）。比较有意思的是，如果没有平方根的操作，算法的效果会变得很差。

由于$G_t$的对角线上包含了关于所有参数$θ$的历史梯度的平方和，现在，我们可以通过$G_t$和$g_t$之间的元素向量乘法$⊙$向量化上述的操作：
$$
\theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{G_{t} + \epsilon}} \odot g_{t}
$$
Adagrad算法的一个主要优点是**无需手动调整学习率**。在大多数的应用场景中，通常采用常数0.01。

Adagrad的一个主要缺点是它在分母中累加梯度的平方：由于没增加一个正项，在整个训练过程中，累加的和会持续增长。这会导致学习率变小以至于最终变得无限小，在学习率无限小时，Adagrad算法将无法取得额外的信息。接下来的算法旨在解决这个不足。



### Adadelta

Adadelta是Adagrad的一种扩展算法，以处理Adagrad学习速率单调递减的问题：不是计算所有的梯度平方，Adadelta将计算历史梯度的窗口大小限制为一个固定值$w$。

在Adadelta中，无需存储先前的$w$个平方梯度，而是将梯度的平方递归地表示成所有历史梯度平方的均值。在$t$时刻的均值$E[g^2]_t$只取决于先前的均值和当前的梯度（分量$γ$类似于动量项）：
$$
E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) g^2_t
$$
将γ设置成与动量项相似的值，即0.9左右。为了简单起见，我们利用参数更新向量$Δθ_t$重新表示SGD的更新过程：
$$
\begin{align} 
\begin{split} 
\Delta \theta_t &= - \eta \cdot g_{t, i} \\ 
\theta_{t+1} &= \theta_t + \Delta \theta_t \end{split} 
\end{align}
$$
我们先前得到的Adagrad参数更新向量变为：
$$
\Delta \theta_t = - \dfrac{\eta}{\sqrt{G_{t} + \epsilon}} \odot g_{t}
$$
现在，我们简单将对角矩阵$G_t$替换成历史梯度的均值$E[g^2]_t$：
$$
\Delta \theta_t = - \dfrac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_{t}
$$
由于分母仅仅是梯度的均方根（root mean squared，RMS）误差，我们可以简写为：
$$
\Delta \theta_t = - \dfrac{\eta}{RMS[g]_{t}} g_t
$$
作者指出上述更新公式中的每个部分单位（与SGD，动量法或者Adagrad中的单位）并不一致，即：更新规则中必须与参数具有相同的假设单位。为了实现这个要求，作者首次定义了另一个指数衰减均值，这次不是梯度平方，而是参数的平方的更新：
$$
E[\Delta \theta^2]_t = \gamma E[\Delta \theta^2]_{t-1} + (1 - \gamma) \Delta \theta^2_t
$$
因此，参数更新的均方根误差为：
$$
RMS[\Delta \theta]_{t} = \sqrt{E[\Delta \theta^2]_t + \epsilon}
$$
由于$RMS[Δθ]_t$是未知的，我们利用参数的均方根误差来近似更新。利用$RMS[Δθ]_{t−1}$替换先前的更新规则中的学习率$η$，最终得到Adadelta的更新规则：
$$
\begin{align} 
\begin{split} 
\Delta \theta_t &= - \dfrac{RMS[\Delta \theta]_{t-1}}{RMS[g]_{t}} g_{t} \\ 
\theta_{t+1} &= \theta_t + \Delta \theta_t 
\end{split} 
\end{align}
$$
使用Adadelta算法，我们甚至都无需设置默认的学习率，因为更新规则中已经移除了学习率。



### RMSprop

RMSprop是一个未被发表的自适应学习率的算法，该算法由Geoff Hinton在其[Coursera课堂的课程6e](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)中提出。

RMSprop和Adadelta在相同的时间里被独立的提出，都起源于对Adagrad的极速递减的学习率问题的求解。实际上，RMSprop是先前我们讨论的Adadelta的第一个更新向量的特例：
$$
\begin{align} 
\begin{split} 
E[g^2]_t &= 0.9 E[g^2]_{t-1} + 0.1 g^2_t \\ 
\theta_{t+1} &= \theta_{t} - \dfrac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_{t} 
\end{split} 
\end{align}
$$
同样，RMSprop将学习率分解成一个平方梯度的指数衰减的平均。Hinton建议将γ设置为0.9，对于学习率η，一个好的固定值为0.001。



### Adam

自适应矩估计（Adaptive Moment Estimation，Adam）是另一种自适应学习率的算法，Adam对每一个参数都计算自适应的学习率。除了像Adadelta和RMSprop一样存储一个指数衰减的历史平方梯度的平均$v_t$，Adam同时还保存一个历史梯度的指数衰减均值$m_t$，类似于动量：
$$
\begin{align} 
\begin{split} 
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\ 
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 
\end{split} 
\end{align}
$$
$m_t$和$v_t$分别是对梯度的一阶矩（均值）和二阶矩（非确定的方差）的估计，正如该算法的名称。当$m_t$和$v_t$初始化为0向量时，Adam的作者发现它们都偏向于0，尤其是在初始化的步骤和当衰减率很小的时候（例如$β_1$和$β_2$趋向于1）。

通过计算偏差校正的一阶矩和二阶矩估计来抵消偏差：
$$
\begin{align} 
\begin{split} 
\hat{m}_t &= \dfrac{m_t}{1 - \beta^t_1} \\ 
\hat{v}_t &= \dfrac{v_t}{1 - \beta^t_2} \end{split} 
\end{align}
$$
正如我们在Adadelta和RMSprop中看到的那样，他们利用上述的公式更新参数，由此生成了Adam的更新规则：
$$
\theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$
作者建议$β_1$取默认值为0.9，$β_2$为0.999，$ϵ$为10−8。他们从经验上表明Adam在实际中表现很好，同时，与其他的自适应学习算法相比，其更有优势。

