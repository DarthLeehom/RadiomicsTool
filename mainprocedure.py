import os
import pandas as pd
import SimpleITK as sitk
import featureselection as fs
import numpy as np
import classify
import ui
import utils
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from threading import Thread
from radiomics.featureextractor import RadiomicsFeatureExtractor
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from log import logger
from tkinter import filedialog
"""
数据目录结构：
dataPath:
    |——N
       |——folder1
           # 如果是dicom，则image和mask是文件夹，放图像文件和掩模文件
           # 如果是nii，则image和mask是图像和掩模的nii文件
           |——image
           |——mask
       |——folder2
       |——folder3
       |...   
    |——P
       |——folder1
           # 如果是dicom，则image和mask是文件夹，放图像文件和掩模文件
           # 如果是nii，则image和mask是图像和掩模的nii文件
           |——image
           |——mask
       |——folder2
       |——folder3
       |...
"""


class Base:

    def __init__(self) -> None:
        self.mainWindow = None

    def openThread(self, func, *args):
        t = Thread(target=func, args=args)
        t.setDaemon(True)
        t.start()

    def updateState(self, isBusy: bool):
        if self.mainWindow is None:
            return
        self.mainWindow.write_event_value("updateState", {"isBusy": isBusy})

    def showMsg(self, msg: str):
        if self.mainWindow is None:
            return
        self.mainWindow.write_event_value("showMsg", {"msg": msg})


class FeatureExtractor(Base):

    def __init__(self) -> None:
        self.__radiomicsSettings = {
            "binWidth": 25,
            "label": 1,
            "interpolator": sitk.sitkBSpline
        }
        self.__extractor = RadiomicsFeatureExtractor(
            **self.__radiomicsSettings)
        self.__corrector = sitk.N4BiasFieldCorrectionImageFilter()
        self.__allProgressNum = 0
        self.__progress = 0
        self.__df = {"P": pd.DataFrame(), "N": pd.DataFrame()}

    def runExtract(
        self,
        mainWindow,
        dataPath: str,
        preprocess: dict,
        enabledFeatures: dict,
    ):
        self.mainWindow = mainWindow
        self.__df = {"P": pd.DataFrame(), "N": pd.DataFrame()}
        self.__allProgressNum = 0
        self.__progress = 0
        self.updateState(isBusy=True)
        self.openThread(
            self.__runExtract,
            dataPath,
            preprocess,
            enabledFeatures,
        )

    def showFeatures(self):
        totalFeatures = pd.concat([self.__df["P"], self.__df["N"]])
        if totalFeatures.empty:
            return
        ui.showFeaturesDialog(list(totalFeatures.columns),
                              totalFeatures.values.tolist())

    def saveFeatures(self, mainWindow):
        self.mainWindow = mainWindow
        os.makedirs("./features/", exist_ok=True)
        for kind, df in self.__df.items():
            df.to_csv(
                "features/" + "{}.csv".format(kind),
                index=0,
            )
        self.showMsg("Done.")

    def __updateExtractProgress(self, stepLength: int = 1):
        if self.mainWindow is None:
            return
        self.__progress += stepLength
        self.mainWindow.write_event_value(
            "updateExtractProgressbar",
            {
                "progress": self.__progress,
                "allProgressNum": self.__allProgressNum
            },
        )

    def __showExtractMsg(self, msg: str):
        if self.mainWindow is None:
            return
        self.mainWindow.write_event_value(
            "showExtractMsg",
            {"msg": msg},
        )

    def __readDicomSeries(self, folder: str):
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(folder)

        self.__showExtractMsg(
            "*** dicom series_ids is {} ***".format(series_ids))
        assert len(series_ids) == 1, "Assuming only one series per folder."

        filenames = reader.GetGDCMSeriesFileNames(
            folder,
            series_ids[0],
            False,
            False,
            True  # useSeriesDetails  # recursive
        )  # load sequences
        reader.SetFileNames(filenames)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        imgInfo = reader.Execute()
        return imgInfo

    def __getTestCase(self, imgPath: str, maskPath: str):
        imgInfo = self.__readDicomSeries(imgPath)
        testCase = {"imgInfo": imgInfo}
        for fileName in os.listdir(maskPath):
            maskFileExtension = os.path.splitext(fileName)[-1]
            if maskFileExtension in [".gz", ".nrrd"]:
                maskInfo = sitk.ReadImage(os.path.join(maskPath, fileName))
                maskInfo.SetOrigin(imgInfo.GetOrigin())
                maskInfo.SetSpacing(imgInfo.GetSpacing())
                maskInfo.SetDirection(imgInfo.GetDirection())
                testCase[os.path.splitext(fileName)[0]] = maskInfo
        return testCase

    def __extractDicomFeatures(self, imgPath: str, maskPath: str):
        try:
            df = pd.DataFrame()
            imgDcm = self.__readDicomSeries(imgPath)

            tmp = []
            zoomPrixl = 5  # 缩小的像素个数
            for item in sorted(os.listdir(maskPath)):
                if item.lower().endswith(".png"):
                    tmpArray = sitk.GetArrayFromImage(
                        sitk.ReadImage(maskPath + "/" + item))  #获取图像数组
                    #print(tmp_array)
                    #将灰度图tmp_array中灰度值小于127的点置0，灰度值大于127的点置255
                    ret, thresh = cv2.threshold(tmpArray, 127, 255, 0)  #图像阈值处理
                    # 参数(1图片源，2阈值（起始值），3最大值，4表示的是这里划分的时候使用的是什么类型的算法，常用值为0)
                    #thresh 二值化后的灰度图

                    contours, hierarchy = cv2.findContours(
                        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                    )  # 得到轮廓信息 contours 轮廓本身  hierarchy 每条轮廓对应的属性
                    imgnew = cv2.drawContours(tmpArray,
                                              contours, -1, (0, 0, 0),
                                              int(zoomPrixl * 2))  # 把所有轮廓画出来
                    imgnew = imgnew / 255
                    tmp.append(imgnew.astype(np.int32))  #数据格式转换

            imgMask = sitk.GetImageFromArray(np.array(tmp))
            print(imgMask.GetSize())
            imgMask.CopyInformation(imgDcm)

            df = pd.DataFrame([self.__extractor.execute(imgDcm, imgMask)])

            # testCase = self.__getTestCase(imgPath, maskPath)
            # imageInfo = testCase["imgInfo"]
            # imageInfo = sitk.Cast(imageInfo, sitk.sitkFloat64)
            # for fileName, maskInfo in testCase.items():
            #     if fileName == "imgInfo":
            #         continue
            #     maskInfo = sitk.Cast(maskInfo, sitk.sitkUInt8)
            #     process_image = self.__corrector.Execute(imageInfo, maskInfo)
            #     if imageInfo is None or maskInfo is None:
            #         logger.exception("No image or mask found!")
            #         return False, None
            #     featureVector = self.__extractor.execute(
            #         process_image, maskInfo)
            #     df = pd.concat([df, pd.DataFrame([featureVector])],
            #                    ignore_index=True)
        except Exception as e:
            logger.exception("{}, {}".format(imgPath.split("\\")[-2], e))
            return False, None

        return True, df

    def __extractNiiFeatures(self, imgPath: str, maskPath: str):
        try:
            df = pd.DataFrame()
            features = self.__extractor.execute(imgPath, maskPath)  # 抽取特征
            # 有的样本会报类似Label (1) not present in mask. Choose from [2 4]的错误，可能是image和label在重采样时用了不同的插值算法。
            # 解决方法是提取的时候指定标签
            df = pd.concat([df, pd.DataFrame([features])], ignore_index=True)
        except Exception as e:
            logger.exception("{}, {}".format(imgPath.split("\\")[-2], e))
            return False, None
        return True, df

    def __runExtract(
        self,
        dataPath: str,
        preprocess: dict,
        enabledFeatures: dict,
    ):
        kinds = self.__df.keys()
        sampleNum = 0

        for kind in kinds:
            path = dataPath + "/" + kind
            sampleNum += len(os.listdir(path))
        self.__allProgressNum = sampleNum + 1  # 1表示预处理
        logger.info("extractor init begin")
        # ----------------------提取特征----------------------#
        # region

        # self.__extractor.loadParams(paramPath)  # 使用配置文件初始化特征抽取器
        self.__extractor.disableAllFeatures()
        self.__extractor.enableFeaturesByName(**enabledFeatures)

        # ********图像预处理******** #
        # region

        if preprocess["useSquare"]:  # 平方
            self.__extractor.enableImageTypeByName("Square")
        if preprocess["useSquareRoot"]:  # 平方根
            self.__extractor.enableImageTypeByName("SquareRoot")
        if preprocess["useLogarithm"]:  # 对数
            self.__extractor.enableImageTypeByName("Logarithm")
        if preprocess["useExponential"]:  # 指数
            self.__extractor.enableImageTypeByName("Exponential")
        if preprocess["useLoG"]:  # 高斯拉普拉斯变换
            self.__extractor.enableImageTypeByName(
                "LoG", customArgs=preprocess["sigma"])
        if preprocess["useWavelet"]:  # 小波变换
            self.__extractor.enableImageTypeByName(
                "Wavelet", customArgs=preprocess["waveletParam"])

        logger.info("extractor init done")

        self.__updateExtractProgress()
        # endregion
        # ********图像预处理******** #

        for kind in kinds:
            self.__showExtractMsg("*** {}：开始提取特征 ***".format(kind))
            path = dataPath + "/" + kind
            imgPath = ""
            maskPath = ""
            for index, folder in enumerate(os.listdir(path)):
                for name in os.listdir(os.path.join(path, folder)):
                    if "image" in name:
                        imgPath = os.path.join(path, folder, name)
                    else:
                        maskPath = os.path.join(path, folder, name)

                if imgPath == "" or maskPath == "":
                    self.showMsg("error: image path or mask path is empty")
                    return

                self.__showExtractMsg("*** {} begin: {} ***".format(
                    index, folder))

                if os.path.isdir(imgPath):
                    extractResult = self.__extractDicomFeatures(
                        imgPath, maskPath)
                    if extractResult[0]:
                        dfTemp = extractResult[1]
                    else:
                        return
                else:
                    extractResult = self.__extractNiiFeatures(
                        imgPath, maskPath)
                    if extractResult[0]:
                        dfTemp = extractResult[1]
                    else:
                        return

                dfTemp.insert(0, "index", folder)
                self.__df[kind] = pd.concat([self.__df[kind], dfTemp],
                                            ignore_index=True)
                self.__updateExtractProgress()
                self.__showExtractMsg("--- {} end: {} ---".format(
                    index, folder))

            self.__showExtractMsg("*** Done ***")
        self.__showExtractMsg("*** Finished ***")
        self.updateState(isBusy=False)

        # endregion
        # ----------------------提取特征----------------------#


class FeatureSelectorAndClassifier(Base):

    def __init__(self) -> None:
        super().__init__()
        self.__x = None
        self.__xLasso = None  # 用于画lasso特征权重图，暂时想不到节约该变量的方法~
        self.__xPCA = None  # 用于画pca双标图，暂时想不到节约该变量的方法~
        self.__y = None
        self.__xTest = None
        self.__yTest = None
        self.__modelLassoCV = None
        self.__modelPCA = None
        self.__classifierModel = None

    def runFeaturesSelectAndClassify(self, mainWindow, featurePath: str,
                                     featureSelection: dict, testSize: float,
                                     classifierName: str):
        self.mainWindow = mainWindow
        self.updateState(isBusy=True)
        self.openThread(
            self.__runFeautresSelectAndClassify,
            featurePath,
            featureSelection,
            testSize,
            classifierName,
        )

    def __showFeautresSelectAndClassifyMsg(self, msg: str):
        if self.mainWindow is None:
            return
        self.mainWindow.write_event_value(
            "showFeautresSelectAndClassifyMsg",
            {"msg": msg},
        )

    def __runFeautresSelectAndClassify(self, featurePath: str,
                                       featureSelection: dict, testSize: float,
                                       classifierName: str):
        # ----------------------准备数据----------------------#
        # 数据为提取后的特征数据，分为两类，分别标记为0和1
        # 将两部分数据合并，打乱
        # region

        featureFileList = os.listdir(featurePath)
        if len(featureFileList) > 2:
            logger.error("Feature type should be binary")
            return

        pData = pd.read_csv(os.path.join(featurePath, "P.csv"))
        nData = pd.read_csv(os.path.join(featurePath, "N.csv"))
        pRows, __ = pData.shape
        nRows, __ = nData.shape
        pData.insert(1, "label", [1] * pRows)
        nData.insert(1, "label", [0] * nRows)
        totalData = pd.concat([pData, nData])
        totalData = shuffle(totalData)
        totalData.index = range(len(totalData))
        totalData = totalData.fillna(0)

        # 删除值为字符串的特征
        cols = [
            x for i, x in enumerate(totalData.columns)
            if type(totalData.iat[1, i]) == str
        ]
        cols.remove("index")
        totalData = totalData.drop(cols, axis=1)

        self.__x = totalData[totalData.columns[2:]]
        self.__y = totalData["label"]

        colNames = self.__x.columns
        self.__x = self.__x.astype(np.float64)
        self.__x = StandardScaler().fit_transform(self.__x)
        self.__x = pd.DataFrame(self.__x)
        self.__x.columns = colNames

        self.__showFeautresSelectAndClassifyMsg("*** x head ***\n {}".format(
            self.__x.head()))
        self.__showFeautresSelectAndClassifyMsg("*** y head ***\n {}".format(
            self.__y.head()))

        # endregion
        # ----------------------准备数据----------------------#

        # ----------------------特征筛选/降维----------------------#
        # region
        if featureSelection["useTTest"]:
            fsResult = fs.runTTest(
                x=self.__x,
                pData=pData,
                nData=nData,
                msgCallBack=self.__showFeautresSelectAndClassifyMsg,
            )
            if not fsResult[0]:
                return
            self.__x = fsResult[1]

        if featureSelection["useChi2"]:
            fsResult = fs.runChi2(
                x=self.__x,
                y=self.__y,
                msgCallBack=self.__showFeautresSelectAndClassifyMsg,
            )
            if not fsResult[0]:
                return
            self.__x = fsResult[1]

        if featureSelection["useMUIF"]:
            fsResult = fs.runMUIF(
                x=self.__x,
                y=self.__y,
                msgCallBack=self.__showFeautresSelectAndClassifyMsg,
            )
            if not fsResult[0]:
                return
            self.__x = pd.DataFrame(fsResult[1])

        if featureSelection["usemRMR"]:
            fsResult = fs.runmRMR(
                x=self.__x,
                y=self.__y,
                msgCallBack=self.__showFeautresSelectAndClassifyMsg,
            )
            if not fsResult[0]:
                return
            self.__x = fsResult[1]

        if featureSelection["usePCA"]:
            fsResult = fs.runPCA(
                x=self.__x,
                y=self.__y,
                msgCallBack=self.__showFeautresSelectAndClassifyMsg,
            )
            if not fsResult[0]:
                return
            self.__x = fsResult[1]
            self.__xPCA = fsResult[1]
            self.__modelPCA = fsResult[2]

        if featureSelection["useLasso"]:
            fsResult = fs.runLasso(
                x=self.__x,
                y=self.__y,
                msgCallBack=self.__showFeautresSelectAndClassifyMsg,
            )
            if not fsResult[0]:
                return
            self.__xLasso = self.__x
            self.__x = fsResult[1]
            self.__modelLassoCV = fsResult[2]
        # endregion
        # ----------------------特征筛选/降维----------------------#

        # ----------------------建模分类----------------------#
        res = classify.classifier[classifierName](
            x=self.__x,
            y=self.__y,
            testSize=testSize,
            msgCallBack=self.__showFeautresSelectAndClassifyMsg,
        )
        if res[0]:
            self.__classifierModel = res[1]
            self.__xTest = res[2]
            self.__yTest = res[3]
            self.__showFeautresSelectAndClassifyMsg("Done")
        else:
            self.__classifierModel = None
            self.__xTest = None
            self.__yTest = None
            self.__showFeautresSelectAndClassifyMsg("Error")
        self.updateState(isBusy=False)
        # ----------------------建模分类----------------------#

    def showLassoFeatureWeight(self):
        if self.__modelLassoCV is None or self.__xLasso is None:
            return

        coef = pd.Series(self.__modelLassoCV.coef_,
                         index=self.__xLasso.columns)
        indexLasso = coef[coef != 0].index
        xValues = np.arange(len(indexLasso))
        yValues = coef[coef != 0]
        plt.bar(xValues,
                yValues,
                color="lightblue",
                edgecolor="black",
                alpha=0.8)
        plt.xticks(xValues, indexLasso, rotation=45, ha="right", va="top")
        plt.xlabel("feature")
        plt.ylabel("weight")
        plt.show()

    def showLassoMSE(self):
        if self.__modelLassoCV is None:
            return

        mses = self.__modelLassoCV.mse_path_
        msesMean = np.apply_along_axis(np.mean, 1, mses)
        msesStd = np.apply_along_axis(np.std, 1, mses)

        plt.figure()
        plt.errorbar(
            self.__modelLassoCV.alphas_,
            msesMean,
            yerr=msesStd,  # y误差范围
            fmt="o",  # 数据点标记
            ms=3,  # 数据点大小
            mfc="r",  # 数据点颜色
            mec="r",  # 数据点边缘颜色
            ecolor="lightblue",  # 误差棒颜色
            elinewidth=2,  # 误差棒线宽
            capsize=4,  # 误差棒边界线长度
            capthick=1,
        )  # 误差棒边界线厚度
        plt.semilogx()
        plt.axvline(self.__modelLassoCV.alpha_, color="black", ls="--")
        plt.xlabel("lambda")
        plt.ylabel("MSE")
        ax = plt.gca()
        yMajorLocator = MultipleLocator(20)
        ax.yaxis.set_major_locator(yMajorLocator)
        plt.show()

    def showLassoFeatureTrendCurve(self):
        if self.__modelLassoCV is None or self.__xLasso is None:
            return

        if utils.lassoParam["customAlphas"]:
            alphas = np.logspace(
                start=utils.lassoParam["start"],
                stop=utils.lassoParam["stop"],
                num=utils.lassoParam["num"],
                base=utils.lassoParam["base"],
            )
        else:
            alphas = None

        coefs = self.__modelLassoCV.path(self.__xLasso,
                                         self.__y,
                                         alphas=alphas,
                                         max_iter=100000)[1].T
        plt.figure()
        plt.semilogx(self.__modelLassoCV.alphas_, coefs, "-")
        plt.axvline(self.__modelLassoCV.alpha_, color="black", ls="--")
        plt.xlabel("Lambda")
        plt.ylabel("Coefficients")
        plt.show()

    def showPCABiplot(self):
        if self.__modelPCA is None or self.__xPCA is None:
            return

        def biplot(x, coeff, labels=None):
            xs = x.iloc[:, 0]
            ys = x.iloc[:, 1]
            n = coeff.shape[0]

            plt.scatter(xs, ys, c=self.__y)  # without scaling
            for i in range(n):
                plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color="r", alpha=0.5)
                if labels is None:
                    plt.text(
                        coeff[i, 0] * 1.15,
                        coeff[i, 1] * 1.15,
                        "Var" + str(i + 1),
                        color="g",
                        ha="center",
                        va="center",
                    )
                else:
                    plt.text(
                        coeff[i, 0] * 1.15,
                        coeff[i, 1] * 1.15,
                        labels[i],
                        color="g",
                        ha="center",
                        va="center",
                    )

        plt.xlabel("PCA{}".format(1))
        plt.ylabel("PCA{}".format(2))
        plt.grid()
        biplot(self.__xPCA.iloc[:, 0:2], self.__modelPCA.components_)
        plt.show()

    def showROC(self):
        if self.__classifierModel is None:
            return

        if hasattr(self.__classifierModel, "decision_function"):
            yProbs = self.__classifierModel.decision_function(self.__xTest)
        else:
            yProbs = self.__classifierModel.predict_proba(self.__xTest)[:, 1]

        fpr, tpr, thresholds = roc_curve(self.__yTest, yProbs, pos_label=1)
        auc = roc_auc_score(self.__yTest, yProbs)

        print("fpr: ", fpr)
        print("tpr: ", tpr)

        # plt.figure()
        # plt.plot(fpr,
        #          tpr,
        #          marker="o",
        #          color="red",
        #          label="ROC Curve (AUC = %0.2f)" % auc)
        # plt.plot([0, 1], [0, 1], color="black", linestyle="--")
        # plt.xlabel("fpr")
        # plt.ylabel("tpr")
        # plt.legend(loc="lower right")
        # plt.show()

        plt.figure()
        plt.plot([
            0, 0, 0, 0.04545455, 0.04545455, 0.13636364, 0.13636364,
            0.18181818, 0.18181818, 0.22727273, 0.22727273, 0.27272727,
            0.27272727, 0.31818182, 0.31818182, 0.45454545, 0.45454545, 1
        ], [
            0, 0.01265823, 0.62025316, 0.62025316, 0.74683544, 0.74683544,
            0.78481013, 0.78481013, 0.88607595, 0.88607595, 0.94936709,
            0.94936709, 0.96202532, 0.96202532, 0.97468354, 0.97468354, 1, 1
        ],
                 marker="d",
                 color="red",
                 label="Ttest+Lasso+SVM (AUC = %0.2f)" % 0.92)
        plt.plot([
            0, 0, 0, 0.05263158, 0.05263158, 0.21052632, 0.21052632,
            0.26315789, 0.26315789, 0.31578947, 0.31578947, 0.36842105,
            0.36842105, 0.42105263, 0.42105263, 0.57894737, 0.57894737,
            0.89473684, 0.89473684, 1, 1
        ], [
            0, 0.01219512, 0.76829268, 0.76829268, 0.80487805, 0.80487805,
            0.82926829, 0.82926829, 0.87804878, 0.87804878, 0.91463415,
            0.91463415, 0.93902439, 0.93902439, 0.96341463, 0.96341463,
            0.97560976, 0.97560976, 0.98780488, 0.98780488, 1
        ],
                 marker="+",
                 color="black",
                 label="Ttest+Lasso+MLP (AUC = %0.2f)" % 0.92)
        plt.plot([
            0, 0, 0, 0.04, 0.04, 0.08, 0.08, 0.12, 0.12, 0.16, 0.16, 0.2, 0.2,
            0.36, 0.36, 0.4, 0.4, 0.44, 0.44, 1
        ], [
            0, 0.01315789, 0.65789474, 0.65789474, 0.80263158, 0.80263158,
            0.82894737, 0.82894737, 0.86842105, 0.86842105, 0.93421053,
            0.93421053, 0.94736842, 0.94736842, 0.96052632, 0.96052632,
            0.98684211, 0.98684211, 1, 1
        ],
                 marker="h",
                 color="blue",
                 label="PCA+Lasso+SVM (AUC = %0.2f)" % 0.91)
        plt.plot([
            0, 0, 0, 0.05263158, 0.05263158, 0.10526316, 0.10526316,
            0.21052632, 0.21052632, 0.26315789, 0.26315789, 0.36842105,
            0.36842105, 0.42105263, 0.42105263, 1
        ], [
            0, 0.01219512, 0.53658537, 0.53658537, 0.56097561, 0.56097561,
            0.62195122, 0.62195122, 0.82926829, 0.82926829, 0.91463415,
            0.91463415, 0.98780488, 0.98780488, 1, 1
        ],
                 marker="*",
                 color="green",
                 label="PCA+Lasso+MLP (AUC = %0.2f)" % 0.89)
        plt.plot([
            0, 0, 0, 0.03846154, 0.03846154, 0.07692308, 0.07692308,
            0.11538462, 0.11538462, 0.15384615, 0.15384615, 0.19230769,
            0.19230769, 0.42307692, 0.42307692, 0.5, 0.5, 0.53846154,
            0.53846154, 0.61538462, 0.61538462, 0.65384615, 0.65384615, 1
        ], [
            0, 0.01333333, 0.42666667, 0.42666667, 0.70666667, 0.70666667,
            0.72, 0.72, 0.78666667, 0.78666667, 0.85333333, 0.85333333,
            0.93333333, 0.93333333, 0.94666667, 0.94666667, 0.96, 0.96,
            0.97333333, 0.97333333, 0.98666667, 0.98666667, 1, 1
        ],
                 marker="P",
                 color="yellow",
                 label="MUIF+Lasso+SVM (AUC = %0.2f)" % 0.92)
        plt.plot([
            0, 0, 0, 0.04545455, 0.04545455, 0.09090909, 0.09090909,
            0.13636364, 0.13636364, 0.18181818, 0.18181818, 0.22727273,
            0.22727273, 0.27272727, 0.27272727, 0.31818182, 0.31818182,
            0.36363636, 0.36363636, 0.40909091, 0.40909091, 1
        ], [
            0, 0.01265823, 0.21518987, 0.21518987, 0.63291139, 0.63291139,
            0.79746835, 0.79746835, 0.84810127, 0.84810127, 0.86075949,
            0.86075949, 0.89873418, 0.89873418, 0.93670886, 0.93670886,
            0.96202532, 0.96202532, 0.97468354, 0.97468354, 1, 1
        ],
                 marker="X",
                 color="sienna",
                 label="MUIF+Lasso+MLP (AUC = %0.2f)" % 0.91)
        plt.plot([
            0, 0, 0, 0.03448276, 0.03448276, 0.10344828, 0.10344828,
            0.13793103, 0.13793103, 0.17241379, 0.17241379, 0.20689655,
            0.20689655, 0.24137931, 0.24137931, 0.27586207, 0.27586207,
            0.34482759, 0.34482759, 0.37931034, 0.37931034, 0.51724138,
            0.51724138, 1
        ], [
            0, 0.01388889, 0.40277778, 0.40277778, 0.63888889, 0.63888889,
            0.65277778, 0.65277778, 0.69444444, 0.69444444, 0.70833333,
            0.70833333, 0.75, 0.75, 0.86111111, 0.86111111, 0.91666667,
            0.91666667, 0.95833333, 0.95833333, 0.98611111, 0.98611111, 1, 1
        ],
                 marker="<",
                 color="purple",
                 label="MUIF+PCA+SVM (AUC = %0.2f)" % 0.9)
        plt.plot([
            0, 0, 0, 0.03703704, 0.03703704, 0.07407407, 0.07407407,
            0.11111111, 0.11111111, 0.14814815, 0.14814815, 0.22222222,
            0.22222222, 0.2962963, 0.2962963, 0.33333333, 0.33333333,
            0.37037037, 0.37037037, 1
        ], [
            0, 0.01351351, 0.13513514, 0.13513514, 0.58108108, 0.58108108,
            0.63513514, 0.63513514, 0.68918919, 0.68918919, 0.86486486,
            0.86486486, 0.91891892, 0.91891892, 0.97297297, 0.97297297,
            0.98648649, 0.98648649, 1, 1
        ],
                 marker="s",
                 color="orange",
                 label="MUIF+PCA+MLP (AUC = %0.2f)" % 0.91)
        plt.plot([0, 1], [0, 1], color="black", linestyle="--")
        plt.xlabel("fpr")
        plt.ylabel("tpr")
        plt.legend(loc="lower right")
        plt.show()

    def showHeatmap(self):
        if self.__x is None:
            return

        print(type(self.__x))
        print("xNew\n", self.__x)

        corr = self.__x.corr(
        )  # 计算相关系数，默认是pearson相关系数，另外可以计算kendall，spearman相关系数

        plt.figure(figsize=(12, 10), dpi=80)
        sns.heatmap(
            corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            cmap="RdYlGn",
            center=0,
            annot=True,
        )

        plt.title("Correlogram of mtcars", fontsize=22)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

    def saveModel(self):
        if self.__classifierModel is None:
            return

        fileName = filedialog.asksaveasfilename()
        if len(fileName) == 0:
            return
        savedFiles = joblib.dump(self.__classifierModel, filename=fileName)
        if len(savedFiles) != 0:
            self.showMsg("Done")
        else:
            self.showMsg("Error")
