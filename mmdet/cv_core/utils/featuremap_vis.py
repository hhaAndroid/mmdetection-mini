import torch
from collections import OrderedDict
from .misc import get_names_dict
from enum import Enum
import numpy as np


class _ForwardType(Enum):
    HOOK = 0
    FORWARD = 1


class ModelOutputs(object):
    def __init__(self, net, summary):
        self._net = net
        self._summary = summary
        self.gradients = []
        self.feature = []

    def reset(self):
        self.gradients = []
        self.feature = []

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def get_gradients(self):
        return self.gradients if len(self.gradients) <= 1 else [self.gradients[-1]]

    def save_forward(self, module, input, output):
        self.feature.append(output)

    def __call__(self, x, index=[-1], vis=False, save_gradient_flag=True):
        self.reset()
        handles = []
        for i in index:
            if i < 0:
                i = len(list(self._summary.keys())) + i
            if vis:
                print(list(self._summary.keys())[i])
            m = self._summary.get(list(self._summary.keys())[i])
            handles.append(m.register_forward_hook(self.save_forward))
            if save_gradient_flag:
                handles.append(m.register_backward_hook(self.save_gradient))
        output = self._net(x)
        # 移除hook
        for handle in handles:
            handle.remove()
        # 用于对付特殊的relu
        feature_map = self.feature if len(self.feature) <= 1 else [self.feature[-1]]
        return feature_map, output


class BaseActivationMapping(object):
    def __init__(self, net, use_gpu=True):
        self._net = net
        self._use_gpu = use_gpu
        self._style = None
        self._summary = None
        self._hooks = None

    def set_hook_style(self, num_channel, input_shape, print_summary=True):
        self._num_channel = num_channel
        self._input_shape = input_shape
        self._print_model_structure(print_summary)

    def set_forward_style(self, forward_func):
        raise NotImplementedError

    def run(self, img, feature_index=1, target=None, activate_fun='softmax'):
        raise NotImplementedError

    def _print_model_structure(self, print_summary=True):
        import torchsummaryX as summaryX
        self._net.apply(self._add_model_forward(get_names_dict(self._net)))
        extra = torch.zeros((2, self._num_channel, self._input_shape[0], self._input_shape[1]))
        if self._use_gpu:
            extra = extra.cuda()
        with torch.no_grad():
            self._net(extra)
        # 删除hook，防止影响
        for handle in self._hooks:
            handle.remove()
        if print_summary:
            summaryX.summary(self._net, extra)

    def _add_model_forward(self, names_dict):
        _summary = OrderedDict()
        hooks = []
        self._summary = _summary
        self._hooks = hooks

        def register_hook(module):
            def hook(module, inputs, outputs):
                module_idx = len(_summary)
                for name, item in names_dict.items():
                    if item == module:
                        key = "{}_{}".format(module_idx, name)
                _summary[key] = module

            if not module._modules:
                hooks.append(module.register_forward_hook(hook))

        return register_hook


class FeatureMapVis(BaseActivationMapping):
    def __init__(self, net, use_gpu=True):
        super(FeatureMapVis, self).__init__(net, use_gpu)

    def set_hook_style(self, num_channel, input_shape, print_summary=True, post_process_func=None):
        super().set_hook_style(num_channel, input_shape, print_summary)
        self._style = _ForwardType.HOOK
        self._post_process_func = post_process_func
        self._model_out = ModelOutputs(self._net, self._summary)

    def set_forward_style(self, forward_func):
        self._forward_func = forward_func
        self._style = _ForwardType.FORWARD

    def run(self, img, feature_index=1, target=None, activate_fun='softmax'):
        assert self._style is not None, 'You need to select the run mode,' \
                                        'you must call set_hook_style() or set_forward_style() one of them'

        data = np.copy(img)
        if self._use_gpu:
            data = torch.from_numpy(np.array([data])).cuda()
        else:
            data = torch.from_numpy(np.array([data]))
        data = data.permute(0, 3, 1, 2)  # 通道在前
        if self._style == _ForwardType.HOOK:
            feature_map, output = self._model_out(data, [feature_index], save_gradient_flag=False)
            if self._post_process_func is not None:
                feature_map, _ = self._post_process_func(feature_map, output)
        elif self._style == _ForwardType.FORWARD:
            feature_map = self._forward_func(data)
        else:
            raise NotImplementedError
        return feature_map