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

