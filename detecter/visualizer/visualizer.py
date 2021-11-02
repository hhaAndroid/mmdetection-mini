import numpy as np
from enum import Enum, unique
import cv2
import colorsys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .builder import VISUALIZERS
import numpy as np
import mmcv
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

__all__ = ['DataType', 'DetVisualizer']

EPS = 1e-2
_SMALL_OBJECT_AREA_THRESH = 1000


# 用于区分预测数据还是标注数据
@unique
class DataType(Enum):
    GT = 'gt'
    PRED = 'pred'


class BaseVisualizer:
    task_dict = {}

    def __init__(self, image=None, img_metas=None, metadata=None, scale=1.0):
        # 预置的各种元数据，例如类名，预定义的任务中meta字段是提前定义好的，不能更改
        self.metadata = metadata
        self.scale = scale
        if image is not None:
            self._setup_fig(image, img_metas)

    def _setup_fig(self, image, img_metas):
        self.img_metas = img_metas
        self.width, self.height = image.shape[1], image.shape[0]
        self._default_font_size = max(
            np.sqrt(self.height * self.width) // 90, 10 // self.scale
        )

        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        img = image.astype("uint8")
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    # 可选 方便后续链式调用
    def set_image(self, image, img_metas=None):
        self._setup_fig(image, img_metas)

    # 获取绘制后真正的数据
    def get_image(self):
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        buffer = np.frombuffer(s, dtype="uint8")
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")

    def save(self, filepath, drawed_image=None):
        if drawed_image is not None:
            mmcv.imwrite(drawed_image, filepath)
        else:
            self.fig.savefig(filepath)

    def show(self, drawed_image=None, winname='window', wait_time=0):
        cv2.namedWindow(winname, 0)
        cv2.imshow(winname, self.get_image() if drawed_image is None else drawed_image)
        cv2.waitKey(wait_time)
        cv2.destroyWindow(winname)

    # -------------基础可视化接口-------------
    def draw_bbox(self, bbox, alpha=0.8, edge_color="g", line_style="-", line_width=1, is_filling=True):
        bbox_x, bbox_y, x1, y1 = bbox
        bbox_w = x1 - bbox_x
        bbox_h = y1 - bbox_y

        linewidth = min(max(line_width, 1), self._default_font_size / 4)

        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y + bbox_h],
                [bbox_x + bbox_w, bbox_y]]

        np_poly = np.array(poly).reshape((4, 2))

        if is_filling:
            p = PatchCollection([Polygon(np_poly)], facecolor=edge_color, linewidths=0, alpha=0.4)
            self.ax.add_collection(p)

        p = PatchCollection([Polygon(np_poly)],
                            facecolor='none',
                            edgecolor=edge_color,
                            linewidth=linewidth * self.scale,
                            alpha=alpha,
                            linestyle=line_style, )
        self.ax.add_collection(p)
        return self

    def draw_text(
            self,
            text,
            position,
            *,
            font_size=None,
            color="g",
            verticalalignment="top",
            horizontalalignment="left",
            rotation=0,
            bbox=None,
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        if bbox is None:
            bbox = dict(facecolor=color, alpha=0.6)

        x, y = position
        self.ax.text(
            x,
            y,
            text,
            size=font_size * self.scale,
            family="sans-serif",
            bbox=bbox,
            verticalalignment=verticalalignment,
            horizontalalignment=horizontalalignment,
            color=color,
        )
        return self

    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None):
        return self

    def draw_circle(self, circle_coord, color, radius=3):
        return self

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        return self

    def draw_binary_mask(
            self, binary_mask, color=None, *, edge_color=None, text=None, alpha=0.5, area_threshold=0
    ):
        return self

    # ------------------------------------------------------------------------
    # 由于其输入的是多个通道，故需要支持参数： 选择激活度最高的topk，然后拼接成一张图显示
    def draw_featmap(self, tensor_chw, topk=-1):
        return self

    # 下游库注册用
    @classmethod
    def register_task(cls, task_name, force=False):
        def _register(task_func):
            if (task_name not in cls.task_dict) or force:
                cls.task_dict[task_name] = task_func
            else:
                raise KeyError(
                    f'{task_name} is already registered in task_dict, '
                    'add "force=True" if you want to override it')
            return task_func

        return _register


@VISUALIZERS.register_module()
class DetVisualizer(BaseVisualizer):
    def __init__(self, image=None, img_metas=None, metadata=None, scale=1.0,
                 gt_bbox_params=None,  # 能够通过配置修改
                 pred_bbox_params=None,
                 class_wise=True):
        super(DetVisualizer, self).__init__(image, img_metas, metadata, scale)
        self._data_sample = None
        # gt 和 pred 可视化配置字典
        self._gt_bbox_params = gt_bbox_params
        self._pred_bbox_params = pred_bbox_params

        # default vis setting
        self._default_gt_bbox_params = dict(edge_color='g')  # 设置默认值
        self._default_pred_bbox_params = dict(edge_color='b')  # 设置默认值
        """gt_vis_setting=dict(bbox_param=dict(color='g'))"""  # 用户配置例子

        self._class_wise = class_wise
        if self.metadata:
            thing_classes = self.metadata.get('thing_classes', [])
        else:
            thing_classes = []

        image2color = dict()
        if len(thing_classes) > 0:
            for clazz in thing_classes:
                image2color[clazz] = (np.random.random((1, 3)) * 0.7 + 0.3).tolist()[0]
        else:
            for clazz_idx in range(5000):
                image2color[str(clazz_idx)] = (np.random.random((1, 3)) * 0.7 + 0.3).tolist()[0]
        self.image2color = image2color

    # ------------------------------------------------------------------------
    # high level api，不支持 batch
    # 如果要支持 batch，建议再新建一个上层接口内部调用 draw 实现，因为涉及到的设置比较多
    def draw(self, data_sample, image=None, img_meta=None, show_gt=True, show_pred=True):
        # 图片绘制对象初始化
        if image is not None:
            self._setup_fig(image, img_meta)

        # 用于跨函数调用时候访问额外数据
        self._data_sample = data_sample

        if show_gt:
            for task in self.task_dict:
                task_attr = 'gt_' + task  # gt 前缀的设置在下游库中预置
                if task_attr in data_sample:
                    self.task_dict[task](self, data_sample[task_attr], DataType.GT)

        if show_pred:
            for task in self.task_dict:
                task_attr = 'pred_' + task
                if task_attr in data_sample:
                    self.task_dict[task](self, data_sample[task_attr], DataType.PRED)
            if 'proposals' in data_sample:
                self.draw_proposals(data_sample.proposals, DataType.PRED)

    @BaseVisualizer.register_task('instances')
    def draw_instance(self, instances, data_type):
        bbox_params = self._default_gt_bbox_params if data_type == DataType.GT else self._default_pred_bbox_params
        if 'bboxes' in instances:
            self._draw_bboxes(instances, bbox_params)
        return self

    @BaseVisualizer.register_task('sem_seg')
    def draw_sem_seg(self, pixel_data, data_type):
        return self

    @BaseVisualizer.register_task('panoptic_seg')
    def draw_panoptic_seg(self, pixel_data, data_type):
        return self

    # 只有预测才有，不需要注册
    def draw_proposals(self, instances, data_type):
        return self

    def _draw_bboxes(self, instances, bbox_params):
        if self.metadata:
            thing_classes = self.metadata.get('thing_classes', [])
        else:
            thing_classes = None
        if self._class_wise:
            for (bbox, label) in zip(instances.bboxes.tensor.numpy(), instances.labels.numpy()):
                clazz = thing_classes[label] if thing_classes else str(label)
                edge_color = self.image2color[clazz]
                bbox_params['edge_color'] = edge_color
                self.draw_bbox(bbox, **bbox_params)

                x0, y0, x1, y1 = bbox
                height_ratio = (y1 - y0) / np.sqrt(self.height * self.width)
                font_size = (
                        np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                        * 0.4
                        * self._default_font_size
                )
                self.draw_text(
                    clazz,
                    (x0, y0),
                    color='white',
                    bbox=dict(facecolor=edge_color, alpha=0.6, pad=1),
                    font_size=font_size,
                )

        else:
            for (bbox, label) in zip(instances.bboxes.tensor.numpy(), instances.labels.numpy()):
                clazz = thing_classes[label]
                self.draw_bbox(bbox, **bbox_params)
                x0, y0, x1, y1 = bbox
                height_ratio = (y1 - y0) / np.sqrt(self.height * self.width)
                font_size = (
                        np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                        * 0.4
                        * self._default_font_size
                )
                self.draw_text(
                    clazz,
                    (x0, y0),
                    color='white',
                    bbox=dict(facecolor=bbox_params['edge_color'], alpha=0.6, pad=1),
                    font_size=font_size,
                )

    # 属性设置
    def set_bbox_params(self, gt_bbox_params, pred_bbox_params):
        pass
