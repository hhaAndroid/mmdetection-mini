import numpy as np


class Kmean(object):
    def __init__(self, cluster_number, number_iter=1, name='iou'):
        self.cluster_number = cluster_number
        self.number_iter = number_iter
        self.name = name

    def _get_distance_measure(self, name='iou'):
        if name == 'iou':
            return self._calc_iou
        else:
            raise NotImplementedError('暂时没有实现')

    def _calc_iou(self, boxes_nx2, clusters_kx2):
        """
        calculate the iou between bboxes and clusters
        Args:
            boxes_nx2(np.ndarray): bboxes's width and height
            clusters_kx2(np.ndarray): clusters_kx2's width and height
        return:
            iou_nxk(np.ndarray): iou between bboxes and clusters
        """
        n = boxes_nx2.shape[0]
        k = self.cluster_number

        box_area = boxes_nx2[:, 0] * boxes_nx2[:, 1]  # 相当于左上角全部移动到0,0点，进行iou计算
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters_kx2[:, 0] * clusters_kx2[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes_nx2[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters_kx2[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes_nx2[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters_kx2[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        iou_nxk = inter_area / (box_area + cluster_area - inter_area)
        return iou_nxk

    def _calc_average_measure(self, boxes_nx2, clusters_kx2):
        """
        calculate the mean iou between bboxes and clusters
        Args:
            boxes_nx2(np.ndarray): bboxes's width and height
            clusters_kx2(np.ndarray): clusters_kx2's width and height
        return:
            mean_iou(np.ndarray): mean iou between boxes and their corresponding clusters
        """
        _distance_measure_fun = self._get_distance_measure(self.name)
        accuracy = np.mean([np.max(_distance_measure_fun(boxes_nx2, clusters_kx2), axis=1)])
        return accuracy

    def _kmeans(self, boxes_nx2):
        """
        cacluate the clusters by kmeans
        Args:
            boxes_nx2(np.ndarray): bboxes's width and height
        would use:
            cluster_number
        would call:
            _calc_iou()
        return:
            clusters(np.ndarray): the anchors for yolo
        """
        k = self.cluster_number
        box_number = boxes_nx2.shape[0]
        last_nearest = np.zeros((box_number,))
        clusters = boxes_nx2[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        _distance_measure_fun = self._get_distance_measure(self.name)
        while True:
            # 距离度量准则是1-iou，iou越大则越近
            distances = 1 - _distance_measure_fun(boxes_nx2, clusters)  # 输出维度 N,k

            current_nearest = np.argmin(distances, axis=1)  # 找出某个点离所有中心最近的索引
            if (last_nearest == current_nearest).all():  # 收敛
                break  # clusters won't change
            for cluster in range(k):  # 更新聚类中心
                if len(boxes_nx2[current_nearest == cluster]) == 0:
                    clusters[cluster] = boxes_nx2[np.random.choice(
                        box_number, 1, replace=False)]
                else:
                    clusters[cluster] = np.median(  # update clusters
                        boxes_nx2[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def clusters(self, wh_data_nx2):
        total_acc = -1
        total_result = []
        for _ in range(self.number_iter):
            result = self._kmeans(wh_data_nx2)  # TODO ga+kmean
            anchor_area = result[:, 0] * result[:, 1]
            area_index = np.argsort(anchor_area)
            result = result[area_index]
            acc = self._calc_average_measure(wh_data_nx2, result) * 100
            if acc > total_acc:
                total_acc = acc
                total_result = result

        # print("K anchors:\n {}".format(total_result.astype(np.int32)))
        print("Accuracy: {:.2f}%".format(total_acc))
        return total_result.astype(np.int32).tolist()
