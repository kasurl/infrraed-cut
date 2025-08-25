import numpy as np
import os
from PIL import Image

__all__ = ['SegmentationMetric']

"""
confusionMetric
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵n*n，初始值全0

    # 像素准确率PA，预测正确的像素/总像素
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    # 类别像素准确率CPA，返回n*1的值，代表每一类，包括背景
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / np.maximum(self.confusionMatrix.sum(axis=1), 1)
        return classAcc

    # 类别平均像素准确率MPA，对每一类的像素准确率求平均
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    # MIoU
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.maximum(np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix), 1)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    # 根据标签和预测图片返回其混淆矩阵
    def genConfusionMatrix(self, imgPredict, imgLabel):
        """生成混淆矩阵：适配512×512灰度图，确保像素值映射到[0, numClass-1]"""
        # 1. 强制转为单通道（灰度图可能被PIL读取为3通道，需降维）
        if len(imgPredict.shape) > 2:
            imgPredict = imgPredict[:, :, 0]  # 取单通道（如RGB转灰度的等效操作）
        if len(imgLabel.shape) > 2:
            imgLabel = imgLabel[:, :, 0]

        # 3. 像素值映射到有效类别（二分类仅允许0和1，避免255等无效值导致混淆矩阵计数错误）
        # 预测图：二值化（>0为目标1，否则为背景0），适配模型输出的二值化结果（如0/255）
        imgPredict = (imgPredict > 0).astype(np.int64)
        # 标签图：同样二值化（确保标签仅含0和1，避免标注工具导出的255等值）
        imgLabel = (imgLabel > 0).astype(np.int64)

        # 4. 过滤无效标签（仅保留[0, numClass-1]的像素）
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        # 计算混淆矩阵的扁平索引（如二分类：0→0*2+0=0，1→0*2+1=1，2→1*2+0=2，3→1*2+1=3）
        label_indices = self.numClass * imgLabel[mask] + imgPredict[mask]
        # 统计每个索引的出现次数（minlength确保结果长度为numClass²，避免漏统计）
        count = np.bincount(label_indices, minlength=self.numClass ** 2)
        # 重塑为numClass×numClass的混淆矩阵
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix


    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    # 更新混淆矩阵
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape  # 确认标签和预测值图片大小相等
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    # 清空混淆矩阵
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))



def evaluate_single_image(predict_path, label_path):
    """单张512×512灰度图的评价（用于调试）"""
    # 读取图像（PIL默认读取为RGB，需转为灰度图）
    imgPredict = Image.open(predict_path).convert('L')  # 'L'模式强制转为单通道灰度图
    imgPredict = np.array(imgPredict)
    imgLabel = Image.open(label_path).convert('L')
    imgLabel = np.array(imgLabel)

    # 初始化评价器（二分类：0=背景，1=目标）
    metric = SegmentationMetric(numClass=2)
    metric.addBatch(imgPredict, imgLabel)

    # 计算指标
    acc = metric.pixelAccuracy()
    macc = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    fwIoU = metric.Frequency_Weighted_Intersection_over_Union()

    # 打印结果
    print(f"单张图像评价结果：")
    print(f"像素准确率(PA)：{acc:.4f}")
    print(f"平均像素准确率(MPA)：{macc:.4f}")
    print(f"平均交并比(mIoU)：{mIoU:.4f}")
    print(f"频权交并比(FW-IoU)：{fwIoU:.4f}")
    return acc, macc, mIoU, fwIoU



def evaluate_batch(pre_dir, label_dir):
    """批量评价512×512灰度图（返回单张指标列表和整体指标）"""
    # 1. 校验目录存在
    assert os.path.exists(pre_dir), f"预测图目录不存在：{pre_dir}"
    assert os.path.exists(label_dir), f"标签图目录不存在：{label_dir}"

    # 2. 获取图像文件名（确保预测图与标签图一一对应，建议文件名一致）
    pre_filenames = sorted([f for f in os.listdir(pre_dir) if f.endswith(('.png', '.jpg', '.bmp'))])
    label_filenames = sorted([f for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg', '.bmp'))])
    assert len(pre_filenames) == len(label_filenames), \
        f"预测图数量({len(pre_filenames)})与标签图数量({len(label_filenames)})不匹配"

    # 3. 初始化指标列表和全局评价器
    acc_list = []
    macc_list = []
    mIoU_list = []
    fwIoU_list = []
    global_metric = SegmentationMetric(numClass=2)  # 全局混淆矩阵（累加所有图像）

    # 4. 逐张处理
    for pre_name, label_name in zip(pre_filenames, label_filenames):
        pre_path = os.path.join(pre_dir, pre_name)
        label_path = os.path.join(label_dir, label_name)

        # 读取并转为单通道灰度图
        imgPredict = Image.open(pre_path).convert('L')
        imgPredict = np.array(imgPredict)
        imgLabel = Image.open(label_path).convert('L')
        imgLabel = np.array(imgLabel)

        # 单张图像评价
        single_metric = SegmentationMetric(numClass=2)
        single_metric.addBatch(imgPredict, imgLabel)
        acc_list.append(single_metric.pixelAccuracy())
        macc_list.append(single_metric.meanPixelAccuracy())
        mIoU_list.append(single_metric.meanIntersectionOverUnion())
        fwIoU_list.append(single_metric.Frequency_Weighted_Intersection_over_Union())

        # 更新全局混淆矩阵
        global_metric.addBatch(imgPredict, imgLabel)

        # 打印单张进度
        print(f"已处理：{pre_name} | PA: {acc_list[-1]:.4f}, mIoU: {mIoU_list[-1]:.4f}")

    # 5. 计算整体指标（两种方式：单张平均 vs 全局混淆矩阵）
    avg_acc = np.mean(acc_list)
    avg_macc = np.mean(macc_list)
    avg_mIoU = np.mean(mIoU_list)
    avg_fwIoU = np.mean(fwIoU_list)

    global_acc = global_metric.pixelAccuracy()
    global_macc = global_metric.meanPixelAccuracy()
    global_mIoU = global_metric.meanIntersectionOverUnion()
    global_fwIoU = global_metric.Frequency_Weighted_Intersection_over_Union()

    # 6. 打印最终结果
    print("\n" + "="*50)
    print("批量评价结果汇总（512×512灰度图，二分类）")
    print("="*50)
    print(f"单张平均指标：")
    print(f"PA: {avg_acc:.4f} ({avg_acc*100:.2f}%), MPA: {avg_macc:.4f} ({avg_macc*100:.2f}%)")
    print(f"mIoU: {avg_mIoU:.4f} ({avg_mIoU*100:.2f}%), FW-IoU: {avg_fwIoU:.4f} ({avg_fwIoU*100:.2f}%)")
    print(f"\n全局混淆矩阵指标：")
    print(f"PA: {global_acc:.4f} ({global_acc*100:.2f}%), MPA: {global_macc:.4f} ({global_macc*100:.2f}%)")
    print(f"mIoU: {global_mIoU:.4f} ({global_mIoU*100:.2f}%), FW-IoU: {global_fwIoU:.4f} ({global_fwIoU*100:.2f}%)")
    print("="*50)

    return (acc_list, macc_list, mIoU_list, fwIoU_list), \
           (global_acc, global_macc, global_mIoU, global_fwIoU)


if __name__ == '__main__':
    # -------------------------- 配置参数 --------------------------
    # 预测图目录（512×512灰度图，二值化结果，如0=背景，255=目标）
    PRE_DIR = '../results_binary/'
    # 标签图目录（512×512灰度图，标注结果，如0=背景，255=目标）
    LABEL_DIR = '../dataset/roadcrack/val/masks/'
    # -------------------------- 执行评价 --------------------------
    # 1. （可选）单张图像调试
    # single_pre = os.path.join(PRE_DIR, "test_001.png")  # 替换为实际文件名
    # single_label = os.path.join(LABEL_DIR, "test_001.png")
    # evaluate_single_image(single_pre, single_label)

    # 2. 批量评价（核心功能）
    _, global_metrics = evaluate_batch(PRE_DIR, LABEL_DIR)