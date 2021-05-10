
import torch
import torch.nn as nn
import torch.nn.functional as F


# Facebook官方实现
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    # ce_loss = - log(p_t)
    # fl_loss = -(1-p_t) ** gamma * log(p_t)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

pred = torch.randn((1, 4))
label = torch.randn((1, 4))
loss = sigmoid_focal_loss(pred, label)
print("facebook sigmoid_focal_loss demo:", loss)


###################################################################################################################################################

class FocalLoss:
    def __init__(self, alpha_t=None, gamma=2):
        """
        :param alpha_t: A list of weights for each class
        :param gamma:
        """
        self.alpha_t = torch.tensor(alpha_t) if alpha_t else None
        self.gamma = gamma

    def __call__(self, outputs, targets):

        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        ce_loss = F.cross_entropy(outputs, targets, weight=self.alpha_t, reduction='none')
        focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()  # mean over the batch

        return focal_loss



outputs = torch.tensor([[2, 1, 2, 1],[2.5, 1, 2.5, 1]])
targets = torch.tensor([0, 1])
alpha_t = [0.25, 0.15, 0.3, 0.3]
fl= FocalLoss(alpha_t, 2)
print("4个类别,权重分别为{},focal_loss:".format(alpha_t), fl(outputs, targets))


###################################################################################################################################################


# class focal_loss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
#         """
#         focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
#         步骤详细的实现了 focal_loss损失函数.
#         :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于目标检测算法中抑制背景类 , retainnet中设置为0.25
#         :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
#         :param num_classes:     类别数量
#         :param size_average:    损失计算方式,默认取均值
#         """
#         super(focal_loss, self).__init__()
#         self.size_average = size_average
#         if isinstance(alpha, list):
#             assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
#             print("  Focal_loss alpha = {}, 将对每一类权重进行精细化赋值  ".format(alpha))
#             self.alpha = torch.Tensor(alpha)
#         else:
#             assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
#             print("  Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用  ".format(alpha))
#             self.alpha = torch.zeros(num_classes)
#             self.alpha[0] += alpha
#             self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

#         self.gamma = gamma

#     def forward(self, preds, labels):
#         """
#         focal_loss损失计算
#         :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
#         :param labels:  实际类别. size:[B,N] or [B]
#         :return:
#         """
#         # assert preds.dim()==2 and labels.dim()==1
#         preds = preds.view(-1, preds.size(-1))
#         self.alpha = self.alpha.to(preds.device)
#         preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
#         preds_softmax = torch.exp(preds_logsoft)  # softmax

#         preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
#         preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
#         self.alpha = self.alpha.gather(0, labels.view(-1))
#         loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
#                           preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

#         loss = torch.mul(self.alpha, loss.t())
#         if self.size_average:
#             loss = loss.mean()
#         else:
#             loss = loss.sum()
#         return loss

# print("---------------------------------------------------------------------")
# pred = torch.randn((3, 5))
# label = torch.tensor([2, 3, 4])
# print("pred:->", pred)
# print("label:->", label)

# print("---------------------------------------------------------------------")

# # alpha设定为0.25,对第一类影响进行减弱(目标检测任务中,第一类为背景类)
# loss_fn = focal_loss(alpha=0.25, gamma=2, num_classes=5)
# loss = loss_fn(pred, label)
# print("alpha=0.25:", loss)

# print("---------------------------------------------------------------------")

# # alpha输入列表,分别对每一类施加不同的权重
# loss_fn = focal_loss(alpha=[1, 2, 3, 1, 2], gamma=2, num_classes=5)
# loss = loss_fn(pred, label)
# print("alpha=[1,2,3,1,2]:", loss)
# print("---------------------------------------------------------------------")