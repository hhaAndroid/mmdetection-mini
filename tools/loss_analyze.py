from mmdet.models import build_loss
from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss
import torch
import numpy as np
import math
from matplotlib import pyplot as plt


def show_loss_grad(aux_predict_out, out_losss, out_grads, label, legend_list, extra_str=''):
    plt.subplot(2, 1, 1)
    plt.title("target={},{}".format(label, extra_str))
    plt.xlabel("predict")
    plt.ylabel("loss")
    pp = []
    for loss in out_losss:
        p, = plt.plot(aux_predict_out, loss)
        pp.append(p)
    plt.legend(pp, list(map(str, legend_list)))

    plt.subplot(2, 1, 2)
    plt.xlabel("predict")
    plt.ylabel("grad")
    pp = []
    for grad in out_grads:
        p, = plt.plot(aux_predict_out, grad)
        pp.append(p)
    plt.legend(pp, list(map(str, legend_list)))
    plt.show()


def collect_data(label=1., num=100, interval=(-1, 1), use_gpu=False):
    if use_gpu:
        target = torch.tensor([[label]], device=0).long()
    else:
        target = torch.tensor([[label]]).long()
    data = np.linspace(interval[0], interval[1], num=num)
    predict_out = []
    aux_predict_out = []
    for d in data:
        if use_gpu:
            predict = torch.tensor([[d]], requires_grad=True, device=0)
        else:
            predict = torch.tensor([[d]], requires_grad=True)
        aux_predict = torch.sigmoid(predict)
        predict_out.append(predict)
        aux_predict_out.append(aux_predict)
    return predict_out, target, aux_predict_out


def get_loss_fun(loss_name, **kwargs):
    if loss_name == 'focal_loss':
        return py_sigmoid_focal_loss
    else:
        raise NotImplementedError


def calc_loss_grad(predicts, target, loss_cls_fun, **kwargs):
    out_loss = []
    out_grad = []
    for i, predict in enumerate(predicts):
        # print('target=', target, 'predict=', predict)
        loss = loss_cls_fun(predict, target, **kwargs)
        loss.backward()
        # print('loss=', round(loss.item(), 6), 'grad=', abs(round(predict.grad.item(), 6)))
        out_loss.append(round(loss.item(), 6))
        grad = predict.grad.item()
        if math.isnan(grad):
            grad = 1e24
        out_grad.append(abs(round(grad, 6)))
    return out_loss, out_grad


def demo_focal_loss():
    interval = (-10, 10)
    use_gpu = False
    label = 1.0
    num = 500
    predict_out, target, aux_predict_out = collect_data(label, num, interval, use_gpu)
    loss_fun = get_loss_fun('focal_loss')

    # 不同gamm曲线
    gammas = [0, 0.5, 1.5, 2, 5]  # gamma用于控制难易样本权重，值越大，对分类错误样本梯度越大(难样本权重大)，focal效应越大，这个参数非常关键
    out_losss = []
    out_grads = []
    for gamma in gammas:
        para = dict(gamma=gamma)
        out_loss, out_grad = calc_loss_grad(predict_out, target, loss_fun, **para)
        out_losss.append(out_loss)
        out_grads.append(out_grad)
    show_loss_grad(aux_predict_out, out_losss, out_grads, label, gammas, 'gamma')

    # 不同alpha曲线
    alphas = [0.1, 0.25, 0.5, 0.75, 0.99]  # alpha用于控制正负样本权重，值越大，权重越大
    out_losss = []
    out_grads = []
    for alpha in alphas:
        para = dict(alpha=alpha)
        out_loss, out_grad = calc_loss_grad(predict_out, target, loss_fun, **para)
        out_losss.append(out_loss)
        out_grads.append(out_grad)
    show_loss_grad(aux_predict_out, out_losss, out_grads, label, alphas, 'alpha')


if __name__ == '__main__':
    demo_focal_loss()
