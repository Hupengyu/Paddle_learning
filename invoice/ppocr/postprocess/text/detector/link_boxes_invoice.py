import numpy as np
import cv2


# # 这个输入是四个点的coor格式
# def get_commodity_info(dt_boxes, rec_res):
#
#     return commodity_info


def get_rect_points(text_boxes, txt_set):
    txts = ''
    for txt in txt_set:
        txts = txts + txt
    x1 = np.min(text_boxes[:, 0])
    y1 = np.min(text_boxes[:, 1])
    x2 = np.max(text_boxes[:, 2])
    y2 = np.max(text_boxes[:, 3])
    res = [[x1, y1, x2, y2], txts]
    return res


class BoxesConnector(object):
    def __init__(self, blocks, imageW, imageH, max_dist=None, overlap_threshold=None):
        # print('max_dist', max_dist)
        # print('overlap_threshold', overlap_threshold)
        self.blocks = blocks
        self.rects = []
        self.txts= []
        for block in self.blocks:
            self.rects.append(block[0])
        for block in self.blocks:
            self.txts.append(block[1])
        self.txts = np.array(self.txts)
        self.rects = np.array(self.rects)    # 转换为array
        self.imageW = imageW    # 输入图片的宽度
        self.imageH = imageH    # 输入图片的高度
        self.max_dist = max_dist  # x轴方向上合并框阈值
        self.overlap_threshold = overlap_threshold  # y轴方向上最大重合度
        self.graph = np.zeros((self.rects.shape[0], self.rects.shape[0]))  # 构建一个N*N的图 N等于rects的数量

        self.r_index = [[] for _ in range(imageW)]  # 构建imageW个空列表

        # 改变rects点的格式为对角式
        self.coordinate_normalization()

        # 此处将所有的rects都加入rect_index中
        for index, rect in enumerate(self.rects):  # r_index第rect[0]个元素表示 第index个(数量可以是0/1/大于1)rect的x轴起始坐标等于rect[0]
            if int(rect[0]) < imageW:
                self.r_index[int(rect[0])].append(index)
            else:  # 边缘的框旋转后可能坐标越界
                self.r_index[imageW - 1].append(index)
        # print(self.r_index)

    def coordinate_normalization(self):
        new_rects = []
        for rect in self.rects:
            x_min = min(rect[0][0], rect[1][0], rect[2][0], rect[3][0])
            x_max = max(rect[0][0], rect[1][0], rect[2][0], rect[3][0])
            y_min = min(rect[0][1], rect[1][1], rect[2][1], rect[3][1])
            y_max = max(rect[0][1], rect[1][1], rect[2][1], rect[3][1])
            new_rect = [int(x_min), int(y_min), int(x_max), int(y_max)]
            new_rects.append(new_rect)
        new_rects = np.array(new_rects)
        self.rects = new_rects

    def calc_overlap_for_Yaxis(self, index1, index2):   # 此时的boxes是对角线坐标
        # 计算两个框在Y轴方向的重合度(Y轴错位程度)——两个边的H的交集的长度！！！！
        height1 = self.rects[index1][3] - self.rects[index1][1]
        height2 = self.rects[index2][3] - self.rects[index2][1]
        tl_y_max = max(self.rects[index1][1], self.rects[index2][1])    # 选取两个框中‘上边’在‘下面’的框的左上角的y_coor
        br_y_min = min(self.rects[index1][3], self.rects[index2][3])    # 选取两个框中‘下边’在‘上面’的框的右下角的y_coor
        # if br_y_min - tl_y_max < 0,则右边的框在左边的上方(不应该拼接)，此时Yaxis_overlap=0,一定小于overlap_threshold
        # if 0 < br_y_min - tl_y_max < min_height
        # if br_y_min - tl_y_max = min_height,此时Yaxis_overlap与overlap_threshold无法确定谁大----只要大于0就拼接！！！
        # br_y_min - tl_y_max========两个边的H的交集的长度！！！！
        Yaxis_overlap = max(0, br_y_min - tl_y_max) / max(height1, height2)
        # Yaxis_overlap = max(0, y1 - y0) / max(height1, height2)

        # print('Yaxis_overlap', Yaxis_overlap)
        return Yaxis_overlap

    def get_proposal(self, index):
        rect = self.rects[index]     # [index][1]: txt
        #   rect[0]:左上角x坐标    rect[2]：右下角x坐标

        # **************先把周围的框剔除******************
        # X:   if rect右边的框的左上角坐标的值> 0.5*imageW或者< 0.12*imageW：则不匹配
        if rect[2] + self.max_dist > 0.55 * self.imageW or rect[2] + self.max_dist < 0.12 * self.imageW:
            return -1
        # Y:   if 框在中间区域，则放弃合并(主要是处理货物清单)
        if 0.39 * self.imageH < rect[1] < 0.80 * self.imageH:
            return -1

        # left:rect框的右边的框： 口-->口  口  口
        for left in range(rect[0] + 1, min(self.imageW - 1, rect[2] + self.max_dist)):
            for idx in self.r_index[left]:  # 遍历在阈值max_dist内的所有的框
                # index: 第index个rect(被比较rect)
                # idx: rect框的右边的框： 口-->口  口  口
                if self.calc_overlap_for_Yaxis(index, idx) > self.overlap_threshold:    # X可以了，然后过滤Y不符合的
                    return idx
        return -1

    def sub_graphs_connected(self):
        sub_graphs = []  # 相当于一个堆栈
        for index in range(self.graph.shape[0]):
            # 第index列全为0且第index行存在非0
            if not self.graph[:, index].any() and self.graph[index, :].any():  # 优先级是not > and > or
                v = index
                # print('v', v)
                sub_graphs.append([v])
                # 级联多个框(大于等于2个)
                # print('self.graph[v, :]', self.graph[v, :])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][
                        0]  # np.where(self.graph[v, :])：(array([5], dtype=int64),)  np.where(self.graph[v, :])[0]：[5]
                    # print('v11', v)
                    sub_graphs[-1].append(v)
                    # print('sub_graphs11', sub_graphs)
        return sub_graphs

    def connect_boxes(self, if_txts=True):
        if if_txts:
            # for idx, (box, txt) in enumerate(zip(self.dt_boxes, txts)):
            for idx, _ in enumerate(self.blocks):
                proposal = self.get_proposal(idx)
                # print('idx11', idx)
                # print('proposal', proposal)
                if proposal >= 0:
                    self.graph[idx][proposal] = 1  # 第idx和proposal个框需要合并则置1——此处是把二维的图gragh中待合并box置1！！！

            sub_graphs = self.sub_graphs_connected()  # sub_graphs [[0, 1], [3, 4, 5]]

            # 不参与合并的框单独存放一个子list
            set_element = set([y for x in sub_graphs for y in x])  # {0, 1, 3, 4, 5}
            for idx, _ in enumerate(self.blocks):
                if idx not in set_element:
                    sub_graphs.append([idx])  # [[0, 1], [3, 4, 5], [2]]

            result_blocks = []
            for sub_graph in sub_graphs:  # 根据graph来合并
                rect_set = self.rects[list(sub_graph)]  # [[228  78 238 128],[240  78 258 128]].....
                txt_set = self.txts[list(sub_graph)]
                block_set = get_rect_points(rect_set, txt_set)
                result_blocks.append(block_set)

            # 对角线—->四个点,并将合并后的boxes延长一点点
            np_result_blocks = sorted(np.array(result_blocks), key=lambda x: (x[0][1], x[0][0]))  # 排序方式！！！优先级（y>x）
            # new_result_blocks = []
            new_result_txts = []
            for result_block in np_result_blocks:
                # new_result_rect = []
                # for i in range(4):
                #     new_result_rect.append([])
                #     for j in range(2):
                #         new_result_rect[i].append(0)
                # x_min = result_block[0][0]
                # y_min = result_block[0][1]
                # x_max = result_block[0][2]
                # y_max = result_block[0][3]
                # new_result_rect[0][0] = x_min - 1  # 对框进行膨胀
                # new_result_rect[0][1] = y_min
                # new_result_rect[1][0] = x_max + 3
                # new_result_rect[1][1] = y_min
                # new_result_rect[2][0] = x_max + 3
                # new_result_rect[2][1] = y_max
                # new_result_rect[3][0] = x_min - 1
                # new_result_rect[3][1] = y_max
                # new_result_rect = np.array(new_result_rect)  # 先转为array
                # new_result_rect = new_result_rect.astype(np.float32)  # 转为float32格式,后面要对应的！！！

                # new_result_blocks.append(new_result_rect)

                new_result_txts.append(result_block[1])

            # return np.array(new_result_blocks), new_result_txts
            return new_result_txts
        else:
            for idx, _ in enumerate(self.rects):
                proposal = self.get_proposal(idx)
                # print('idx11', idx)
                # print('proposal', proposal)
                if proposal >= 0:
                    self.graph[idx][proposal] = 1  # 第idx和proposal个框需要合并则置1——此处是把二维的图gragh中待合并box置1！！！

            sub_graphs = self.sub_graphs_connected()  # sub_graphs [[0, 1], [3, 4, 5]]

            # 不参与合并的框单独存放一个子list
            set_element = set([y for x in sub_graphs for y in x])  # {0, 1, 3, 4, 5}
            for idx, _ in enumerate(self.rects):
                if idx not in set_element:
                    sub_graphs.append([idx])  # [[0, 1], [3, 4, 5], [2]]

            result_rects = []
            for sub_graph in sub_graphs:    # 根据graph来合并
                rect_set = self.rects[list(sub_graph)]  # [[228  78 238 128],[240  78 258 128]].....
                rect_set = get_rect_points(rect_set)
                result_rects.append(rect_set)

            # 对角线—->四个点,并将合并后的boxes延长一点点
            new_result_rects = []
            for result_rect in result_rects:
                new_result_rect = []
                for i in range(4):
                    new_result_rect.append([])
                    for j in range(2):
                        new_result_rect[i].append(0)
                x_min = result_rect[0]
                y_min = result_rect[1]
                x_max = result_rect[2]
                y_max = result_rect[3]
                new_result_rect[0][0] = x_min - 3   # 对框进行膨胀
                new_result_rect[0][1] = y_min
                new_result_rect[1][0] = x_max + 3
                new_result_rect[1][1] = y_min
                new_result_rect[2][0] = x_max + 3
                new_result_rect[2][1] = y_max
                new_result_rect[3][0] = x_min - 3
                new_result_rect[3][1] = y_max
                new_result_rect = np.array(new_result_rect)     # 先转为array
                new_result_rect = new_result_rect.astype(np.float32)    # 转为float32格式,后面要对应的！！！

                new_result_rects.append(new_result_rect)

            return np.array(new_result_rects)


if __name__ == '__main__':

    # rects = []
    # rects.append(np.array([228, 78, 238, 128]))
    # rects.append(np.array([240, 78, 258, 128]))
    # rects.append(np.array([241, 130, 259, 140]))
    # rects.append(np.array([79, 76, 127, 130]))
    # rects.append(np.array([130, 76, 150, 130]))
    # rects.append(np.array([152, 78, 172, 131]))
    #
    # rects.append(np.array([79, 150, 109, 180]))

    rects = [[144, 5, 192, 25], [25, 6, 64, 25], [66, 6, 141, 25], [193, 5, 275, 33], [269, 30, 354, 50],
             [26, 30, 182, 52], [185, 28, 265, 55], [25, 56, 89, 76], [93, 56, 229, 78], [232, 56, 262, 76],
             [264, 52, 343, 81]]

    # 创建一个白纸
    show_image = np.zeros([500, 500, 3], np.uint8) + 255

    connector = BoxesConnector(rects, 500, max_dist=15, overlap_threshold=0.2)  # 输入：rects
    new_rects = connector.connect_boxes()
    print(new_rects)

    # for rect in rects:
    #     cv2.rectangle(show_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 1)

    for rect in new_rects:
        cv2.rectangle(show_image, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 1)
    cv2.imshow('res', show_image)
    cv2.waitKey(0)
