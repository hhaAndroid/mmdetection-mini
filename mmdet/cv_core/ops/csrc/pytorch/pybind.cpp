#include "pytorch_cpp_helper.hpp"


void sigmoid_focal_loss_forward(Tensor input, Tensor target, Tensor weight,
                                Tensor output, float gamma, float alpha);

void sigmoid_focal_loss_backward(Tensor input, Tensor target, Tensor weight,
                                 Tensor grad_input, float gamma, float alpha);

void softmax_focal_loss_forward(Tensor input, Tensor target, Tensor weight,
                                Tensor output, float gamma, float alpha);

void softmax_focal_loss_backward(Tensor input, Tensor target, Tensor weight,
                                 Tensor buff, Tensor grad_input, float gamma,
                                 float alpha);

void bbox_overlaps(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                   const int mode, const bool aligned, const int offset);


Tensor nms(Tensor boxes, Tensor scores, float iou_threshold, int offset);

Tensor softnms(Tensor boxes, Tensor scores, Tensor dets, float iou_threshold,
               float sigma, float min_score, int method, int offset);

std::vector<std::vector<int> > nms_match(Tensor dets, float iou_threshold);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sigmoid_focal_loss_forward", &sigmoid_focal_loss_forward,
        "sigmoid_focal_loss_forward ", py::arg("input"), py::arg("target"),
        py::arg("weight"), py::arg("output"), py::arg("gamma"),
        py::arg("alpha"));
  m.def("sigmoid_focal_loss_backward", &sigmoid_focal_loss_backward,
        "sigmoid_focal_loss_backward", py::arg("input"), py::arg("target"),
        py::arg("weight"), py::arg("grad_input"), py::arg("gamma"),
        py::arg("alpha"));
  m.def("softmax_focal_loss_forward", &softmax_focal_loss_forward,
        "softmax_focal_loss_forward", py::arg("input"), py::arg("target"),
        py::arg("weight"), py::arg("output"), py::arg("gamma"),
        py::arg("alpha"));
  m.def("softmax_focal_loss_backward", &softmax_focal_loss_backward,
        "softmax_focal_loss_backward", py::arg("input"), py::arg("target"),
        py::arg("weight"), py::arg("buff"), py::arg("grad_input"),
        py::arg("gamma"), py::arg("alpha"));
  m.def("bbox_overlaps", &bbox_overlaps, "bbox_overlaps", py::arg("bboxes1"),
        py::arg("bboxes2"), py::arg("ious"), py::arg("mode"),
        py::arg("aligned"), py::arg("offset"));
  m.def("nms", &nms, "nms (CPU/CUDA) ", py::arg("boxes"), py::arg("scores"),
        py::arg("iou_threshold"), py::arg("offset"));
  m.def("softnms", &softnms, "softnms (CPU) ", py::arg("boxes"),
        py::arg("scores"), py::arg("dets"), py::arg("iou_threshold"),
        py::arg("sigma"), py::arg("min_score"), py::arg("method"),
        py::arg("offset"));
  m.def("nms_match", &nms_match, "nms_match (CPU) ", py::arg("dets"),
        py::arg("iou_threshold"));

}
