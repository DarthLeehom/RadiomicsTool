import sys
import os
import logging
import pandas as pd
import numpy as np
import SimpleITK as sitk
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from threading import Thread
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    RepeatedKFold,
    GridSearchCV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.feature_selection import (
    mutual_info_classif,
    chi2,
    SelectKBest,
    SelectPercentile,
)
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import pearsonr, ttest_ind, levene
from matplotlib.pyplot import MultipleLocator
from collections import Counter
from radiomics.featureextractor import RadiomicsFeatureExtractor
from xgboost import XGBClassifier
import pymrmr
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

# def thread_it(func, *args):
#     t = Thread(target=func, args=args)
#     t.setDaemon(True)
#     t.start()

# def readDicomSeries(folder: str):
#     reader = sitk.ImageSeriesReader()
#     series_ids = reader.GetGDCMSeriesIDs(folder)

#     print("dicom series_ids is ", series_ids)
#     assert len(series_ids) == 1, "Assuming only one series per folder."

#     filenames = reader.GetGDCMSeriesFileNames(
#         folder,
#         series_ids[0],
#         False,
#         False,
#         True  # useSeriesDetails  # recursive
#     )  # load sequences
#     reader.SetFileNames(filenames)
#     reader.MetaDataDictionaryArrayUpdateOn()
#     reader.LoadPrivateTagsOn()
#     imgInfo = reader.Execute()
#     return imgInfo

# def getTestCase(imgPath: str, maskPath: str):
#     imgInfo = readDicomSeries(imgPath)
#     testCase = {"imgInfo": imgInfo}
#     for fileName in os.listdir(maskPath):
#         maskFileExtension = os.path.splitext(fileName)[-1]
#         if maskFileExtension in [".gz", ".nrrd"]:
#             maskInfo = sitk.ReadImage(os.path.join(maskPath, fileName))
#             maskInfo.SetOrigin(imgInfo.GetOrigin())
#             maskInfo.SetSpacing(imgInfo.GetSpacing())
#             maskInfo.SetDirection(imgInfo.GetDirection())
#             testCase[os.path.splitext(fileName)[0]] = maskInfo
#     return testCase

# def extractDicomFeatures(imgPath: str, maskPath: str):
#     try:
#         df = pd.DataFrame()
#         testCase = getTestCase(imgPath, maskPath)
#         imageInfo = testCase["imgInfo"]
#         imageInfo = sitk.Cast(imageInfo, sitk.sitkFloat64)
#         for fileName, maskInfo in testCase.items():
#             if fileName == "imgInfo":
#                 continue
#             maskInfo = sitk.Cast(maskInfo, sitk.sitkUInt8)
#             process_image = corrector.Execute(imageInfo, maskInfo)
#             if imageInfo is None or maskInfo is None:
#                 logger.exception("No image or mask found!")
#                 sys.exit()
#             featureVector = extractor.execute(process_image, maskInfo)
#             df = pd.concat([df, pd.DataFrame([featureVector])],
#                            ignore_index=True)
#     except Exception as e:
#         logger.exception("{}, {}".format(folder, e))
#         sys.exit()

#     return df

# def extractNiiFeatures(imgPath: str, maskPath: str):
#     try:
#         df = pd.DataFrame()
#         features = extractor.execute(imgPath, maskPath)  # 抽取特征
#         # 有的样本会报类似Label (1) not present in mask. Choose from [2 4]的错误，可能是image和label在重采样时用了不同的插值算法。
#         # 解决方法是提取的时候指定标签
#         df = pd.concat([df, pd.DataFrame([features])], ignore_index=True)
#     except Exception as e:
#         logger.exception("{}, {}".format(folder, e))
#         sys.exit()
#     return df

# from imblearn.over_sampling import SMOTE  # 新版本有bug

# ----------------------配置日志----------------------#
# region

# def loggerConfig(loggerPath: str, loggerName: str):
#     logger = logging.getLogger(loggerName)
#     logger.setLevel(level=logging.DEBUG)
#     filter = logging.Filter("radiomicsTool")
#     formatter = logging.Formatter(
#         "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(lineno)d - %(message)s"
#     )
#     handler = logging.FileHandler(loggerPath, encoding="UTF-8")
#     handler.setLevel(logging.DEBUG)
#     console = logging.StreamHandler()
#     console.setLevel(logging.DEBUG)
#     handler.addFilter(filter)
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.addHandler(console)
#     return logger

# logger = loggerConfig(loggerPath="./radiomicstool.log",
#                       loggerName="radiomicsTool")

# endregion
# ----------------------配置日志----------------------#

# ----------------------提取特征----------------------#
# region

# kinds = ["P", "N"]
# featureInit = {}  # 需要根据具体界面选择，打开需要提取的特征
# featureInit["shape"] = []
# featureInit["firstorder"] = []
# featureInit["glcm"] = []
# featureInit["glrlm"] = []
# featureInit["glszm"] = []
# featureInit["gldm"] = []
# featureInit["ngtdm"] = []
# paraPath = "yaml/Params.yaml"
# extractor = RadiomicsFeatureExtractor(paraPath)  # 使用配置文件初始化特征抽取器
# extractor.disableAllFeatures()
# extractor.enableFeaturesByName(**featureInit)
# corrector = sitk.N4BiasFieldCorrectionImageFilter()
# dataPath = "data/BraTS19/"

# # ********图像预处理******** #
# # region

# square = False
# squareRoot = False
# logarithm = False
# exponential = False
# LoG = False
# sigma = {"sigma": [2, 3, 4, 5]}
# wavelet = False

# # 小波类型总表
# biorList = [
#     1.1, 1.3, 1.5, 2.2, 2.4, 2.6, 2.8, 3.1, 3.3, 3.5, 3.7, 3.9, 4.4, 5.5, 6.8
# ]
# coifList = [i for i in range(1, 6)]
# dbList = [i for i in range(1, 21)]
# symList = [i for i in range(2, 21)]
# waveletTypeList = (["haar", "dmey"] + ["bior{}".format(i) for i in biorList] +
#                    ["rbio{}".format(i) for i in biorList] +
#                    ["coif{}".format(i)
#                     for i in coifList] + ["db{}".format(i) for i in dbList] +
#                    ["sym{}".format(i) for i in symList])

# waveletParam = {"wavelet": "coif1", "start_level": 0, "level": 2}

# if square:  # 平方
#     extractor.enableImageTypeByName("Square")
# if squareRoot:  # 平方根
#     extractor.enableImageTypeByName("SquareRoot")
# if logarithm:  # 对数
#     extractor.enableImageTypeByName("Logarithm")
# if exponential:  # 指数
#     extractor.enableImageTypeByName("Exponential")
# if LoG:  # 高斯拉普拉斯变换
#     extractor.enableImageTypeByName("LoG", customArgs=sigma)
# if wavelet:  # 小波变换
#     extractor.enableImageTypeByName("Wavelet", customArgs=waveletParam)

# endregion
# ********图像预处理******** #

# for kind in kinds:
#     print("*** {}：开始提取特征 ***".format(kind))
#     featureDict = dict()
#     df = pd.DataFrame()
#     path = dataPath + kind
#     for index, folder in enumerate(os.listdir(path)):
#         for name in os.listdir(os.path.join(path, folder)):
#             if "image" in name:
#                 imgPath = os.path.join(path, folder, name)
#             else:
#                 maskPath = os.path.join(path, folder, name)

#         print("*** ", index, " begin: ", folder, " ***")

#         dfTemp = (extractDicomFeatures(imgPath, maskPath)
#                   if os.path.isdir(imgPath) else extractNiiFeatures(
#                       imgPath, maskPath))
#         dfTemp.insert(0, "index", folder)
#         df = pd.concat([df, dfTemp], ignore_index=True)
#         print("--- ", index, " end: ", folder, " ---")

#     # df.columns = featureDict.keys()
#     df.to_csv("features/" + "{}.csv".format(kind), index=0)
#     print("*** Done ***")
# print("*** 完成 ***")

# endregion
# ----------------------提取特征----------------------#

# ----------------------准备数据----------------------#
# 数据为提取后的特征数据，分为两类，分别标记为0和1
# 将两部分数据合并，打乱
# region

# pData = pd.read_csv("features/P.csv")
# nData = pd.read_csv("features/N.csv")
# pRows, __ = pData.shape
# nRows, __ = nData.shape
# pData.insert(1, "label", [1] * pRows)
# nData.insert(1, "label", [0] * nRows)
# totalData = pd.concat([pData, nData])
# totalData = shuffle(totalData)
# totalData.index = range(len(totalData))
# totalData = totalData.fillna(0)

# # 删除值为字符串的特征
# cols = [
#     x for i, x in enumerate(totalData.columns)
#     if type(totalData.iat[1, i]) == str
# ]
# cols.remove("index")
# totalData = totalData.drop(cols, axis=1)

# endregion
# ----------------------准备数据----------------------#

# ----------------------TTest----------------------#
# region

# indexTTest = []
# for colName in totalData.columns[2:]:
#     try:
#         if levene(pData[colName], nData[colName])[1] > 0.05:
#             if ttest_ind(pData[colName], nData[colName],
#                          equal_var=True)[1] < 0.05:
#                 indexTTest.append(colName)
#         else:
#             if ttest_ind(pData[colName], nData[colName],
#                          equal_var=False)[1] < 0.05:
#                 indexTTest.append(colName)
#     except Exception as e:
#         logger.exception("{}, {}".format(colName, e))
#         sys.exit()

# print("*** features num after ttest: ", len(indexTTest), " ***")

# x = totalData[indexTTest]
# y = totalData["label"]
# x = x.apply(pd.to_numeric, errors="ignore")

# colNames = x.columns
# x = x.astype(np.float64)
# x = StandardScaler().fit_transform(x)
# x = pd.DataFrame(x)
# x.columns = colNames

# print("*** x head ***\n", x.head())
# print("*** y head ***\n", y.head())

# endregion
# ----------------------TTest----------------------#

# ----------------------Lasso----------------------#
# region

# print("*** x shape: ", x.shape, " ***")
# alphas = np.logspace(-2, 1, 50)

# try:
#     modelLassoCV = LassoCV(alphas=alphas, cv=10, max_iter=100000).fit(
#         x, y.values.ravel()
#     )
# except Exception as e:
#     logger.exception("{}".format(e))
#     sys.exit()

# print("*** Lasso lambda: ", modelLassoCV.alpha_, " ***")
# coef = pd.Series(modelLassoCV.coef_, index=x.columns)
# print(
#     "*** Lasso picked "
#     + str(sum(coef != 0))
#     + " variables and eliminated the other "
#     + str(sum(coef == 0)),
#     " ***",
# )
# indexLasso = coef[coef != 0].index
# x = x[indexLasso]
# print("*** coef != 0 ***\n", coef[coef != 0])

# endregion
# ----------------------Lasso----------------------#

# ----------------------特征降维PCA----------------------#
# 特征降维和特征筛选的区别：
# 特征降维是找到特征之间的映射关系，通过映射关系将多个特征合并，映射后特征值会发生变化；
# 特征筛选就是单纯的从原始特征中选择部分特征，特征选择前后的值不变
# region

# print("*** x shape: ", x.shape, " ***")

# try:
#     modelPCA = PCA(n_components=0.99)  # 标识降维后包含原来99%的特征
#     modelPCA.fit(x, y)
#     x = modelPCA.transform(x)
# except Exception as e:
#     logger.exception("{}".format(e))
#     sys.exit()
# print("*** PCA variance ratio ***\n", modelPCA.explained_variance_ratio_)
# print("*** x shape: ", x.shape, " ***")
# print("*** x:\n", x, " ***")

# endregion
# ----------------------特征降维PCA----------------------#

# ----------------------MUIF互信息/卡方检验----------------------#
# 互信息：mutual_info_classif
# 卡方检验：chi2
# 值越大，表示相关性越高
# 这里选择特征可以使用sklearn.feature_selection提供的SelectKBest和SelectPercentile
# SelectKBest表示选择K个相关性最高的特征
# SelectPercentile表示按照相关性从高到低，选择多少百分比的特征
# region

# selectMethod = "MUIF"  # MUIF or chi2
# selectPrinciple = "KBest"  # KBest or percentile

# try:
#     if selectMethod == "MUIF" and selectPrinciple == "KBest":
#         x = SelectKBest(mutual_info_classif, k=8).fit_transform(x, y)
#         print("*** SelectKBest: x shape after MUIF: ", x.shape, " ***")
#     elif selectMethod == "MUIF" and selectPrinciple == "percentile":
#         x = SelectPercentile(mutual_info_classif, percentile=55).fit_transform(x, y)
#         print("*** SelectPercentile: x shape after MUIF: ", x.shape, " ***")
#     elif selectMethod == "chi2" and selectPrinciple == "KBest":
#         x = SelectKBest(chi2, k=8).fit_transform(x, y)
#         print("*** SelectKBest: x shape after chi2: ", x.shape, " ***")
#     elif selectMethod == "chi2" and selectPrinciple == "percentile":
#         x = SelectPercentile(chi2, percentile=10).fit_transform(x, y)
#         print("*** SelectPercentile: x shape after chi2: ", x.shape, " ***")
# except Exception as e:
#     logger.exception("{}".format(e))
#     sys.exit()

# endregion
# ----------------------MUIF互信息/卡方检验----------------------#

# ----------------------最大相关最小冗余准则mRMR----------------------#
# 核心思想是从给定的特征集合中寻找与目标量有最大相关性且特征相互之间具有最少冗余性的特征子集。
# 函数一共三个参数：
# 参数一：DataFrame，要求第一列是目标量，其他列是特征量，其中首行必须是特征名称，且必须是字符形式，例如a1,a2，等
# 参数二：选择的方法，有'MID'、'MIQ'两种。MID是基于互信息的mRMR，MIQ是基于商的mRMR。
# 参数三：要求int类型，最后输出的特征数量。
# region

# print("*** totalData head ***\n", totalData.head())
# colnames = totalData.columns.values.tolist()
# colnames.remove("index")
# dataMRMR = totalData[colnames].copy()
# dataMRMR = dataMRMR.astype(np.float64)
# dataMRMR = StandardScaler().fit_transform(dataMRMR)
# dataMRMR = pd.DataFrame(dataMRMR)
# dataMRMR.columns = colnames
# print("*** dataMRMR head ***\n", dataMRMR.head())

# try:
#     mrmr = pymrmr.mRMR(dataMRMR, "MIQ", 8)
# except Exception as e:
#     logger.exception("{}".format(e))
#     sys.exit()

# print("*** mrmr ***\n", mrmr)

# x = dataMRMR[mrmr].copy()

# print("*** x after mRMR ***\n", x.shape, x.head())

# endregion
# ----------------------最大相关最小冗余准则mRMR----------------------#

# ----------------------画图：Lasso特征权重----------------------#
# region

# xValues = np.arange(len(indexLasso))
# yValues = coef[coef != 0]
# plt.bar(xValues, yValues, color="lightblue", edgecolor="black", alpha=0.8)
# plt.xticks(xValues, indexLasso, rotation=45, ha="right", va="top")
# plt.xlabel("feature")
# plt.ylabel("weight")
# plt.show()

# endregion
# ----------------------画图：Lasso特征权重----------------------#

# ----------------------画图：Lasso均方误差----------------------#
# region

# mses = (modelLassoCV.mse_path_)
# msesMean = np.apply_along_axis(np.mean, 1, mses)
# msesStd = np.apply_along_axis(np.std, 1, mses)

# plt.figure()
# plt.errorbar(modelLassoCV.alphas_, msesMean
#              , yerr = msesStd  #y误差范围
#              , fmt = "o"  #数据点标记
#              , ms = 3  #数据点大小
#              , mfc = "r"  #数据点颜色
#              , mec = "r"  #数据点边缘颜色
#              , ecolor = "lightblue"  #误差棒颜色
#              , elinewidth = 2  #误差棒线宽
#              , capsize = 4  #误差棒边界线长度
#              , capthick = 1)  #误差棒边界线厚度
# plt.semilogx()
# plt.axvline(modelLassoCV.alpha_, color = "black", ls = "--")
# plt.xlabel("lambda")
# plt.ylabel("MSE")
# ax = plt.gca()
# yMajorLocator = MultipleLocator(0.05)
# ax.yaxis.set_major_locator(yMajorLocator)
# plt.show()

# endregion
# ----------------------画图：Lasso均方误差----------------------#

# ----------------------画图：Lasso特征随lambda变化曲线----------------------#
# region

# coefs = modelLassoCV.path(x, y, alphas=alphas, max_iter=100000)[1].T
# plt.figure()
# plt.semilogx(modelLassoCV.alphas_, coefs, "-")
# plt.axvline(modelLassoCV.alpha_, color="black", ls="--")
# plt.xlabel("Lambda")
# plt.ylabel("Coefficients")
# plt.show()

# endregion
# ----------------------画图：Lasso特征随lambda变化曲线----------------------#

# ----------------------画图：PCA双标图----------------------#
# region

# def pcaPlot(x, coeff, labels=None):
#     xs = x.iloc[:, 0]
#     ys = x.iloc[:, 1]
#     n = coeff.shape[0]

#     plt.scatter(xs, ys, c=y)  # without scaling
#     for i in range(n):
#         plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color="r", alpha=0.5)
#         if labels is None:
#             plt.text(
#                 coeff[i, 0] * 1.15,
#                 coeff[i, 1] * 1.15,
#                 "Var" + str(i + 1),
#                 color="g",
#                 ha="center",
#                 va="center",
#             )
#         else:
#             plt.text(
#                 coeff[i, 0] * 1.15,
#                 coeff[i, 1] * 1.15,
#                 labels[i],
#                 color="g",
#                 ha="center",
#                 va="center",
#             )

# plt.xlabel("PCA{}".format(1))
# plt.ylabel("PCA{}".format(2))
# plt.grid()
# pcaPlot(x.iloc[:, 0:2], modelPCA.components_)
# plt.show()

# endregion
# ----------------------画图：PCA双标图----------------------#

# ----------------------RandomForest----------------------#
# region

# xTrainRF, xTestRF, yTrainRF, yTestRF = train_test_split(x, y, test_size = 0.3)

# try:
#     modelRF = RandomForestClassifier(n_estimators = 20).fit(xTrainRF, yTrainRF)
# except Exception as e:
#     logger.exception("{}".format(e))
#     sys.exit()

# scoreRF = modelRF.score(xTestRF, yTestRF)
# print("*** scoreRF: ", scoreRF, " ***")

# endregion
# ----------------------RandomForest----------------------#

# ----------------------SVM----------------------#
# region

# xTrainSVM, xTestSVM, yTrainSVM, yTestSVM = train_test_split(x,
#                                                             y,
#                                                             test_size=0.3)

# try:
#     modelSVM = svm.SVC(kernel="rbf", gamma="auto",
#                        probability=True).fit(xTrainSVM, yTrainSVM)
# except Exception as e:
#     logger.exception("{}".format(e))
#     sys.exit()

# scoreSVM = modelSVM.score(xTestSVM, yTestSVM)
# print("*** scoreSVM: ", scoreSVM, " ***")

# endregion
# ----------------------SVM----------------------#

# ----------------------参数优化：SVM----------------------#
# region

# cs = np.logspace(-1, 3, 10, base = 2)
# gammas = np.logspace(-4, 1, 50, base = 2)
# paramGrid = dict(c = cs, gamma = gammas)

# try:
#     grid = GridSearchCV(svm.SVC(kernel = "rbf"), param_grid = paramGrid, cv = 10).fit(x, y)
# except Exception as e:
#     logger.exception("{}".format(e))
#     sys.exit()

# print("*** GridSearchCV besr params ***\n", grid.best_params_)
# c = grid.best_params_["c"]
# gamma = grid.best_params_["gamma"]

# endregion
# ----------------------参数优化：SVM----------------------#

# ----------------------p次k折交叉验证：SVM---------------------#
# region

# rkf = RepeatedKFold(n_splits = 3, n_repeats = 2)
# for trainIndex, testIndex in rkf.split(x):
#     xTrain = x.iloc[trainIndex]
#     xTest = x.iloc[testIndex]
#     yTrain = y.iloc[trainIndex]
#     yTest = y.iloc[testIndex]

#     try:
#         modelSVM = svm.SVC(kernel = "rbf", C=C, gamma = gamma, probability = True).fit(xTrain, yTrain)
#     except Exception as e:
#         logger.exception("trainIndex: {}-testIndex: {}, {}".format(trainIndex, testIndex, e))
#         sys.exit()

#     scoreSVM = modelSVM.score(xTest, yTest)
#     print("*** scoreSVM after repeated k fold: ", scoreSVM, " ***")
# print()

# endregion
# ----------------------p次k折交叉验证：SVM----------------------#

# ----------------------画图：SVM ROC曲线----------------------#
# region

# yProbs = modelSVM.predict_proba(x)
# fpr, tpr, thresholds = roc_curve(y, yProbs[:,1], pos_label = 1)

# plt.plot(fpr, tpr, marker = "o")
# plt.xlabel('fpr')
# plt.ylabel('tpr')
# plt.show()

# aucScore = roc_auc_score(y, modelSVM.predict(x))
# print("*** aucScore: ", aucScore, " ***")

# endregion
# ----------------------画图：SVM ROC曲线----------------------#

# ----------------------神经网络算法：多层感知器----------------------#
# region

# xTrainMLP, xTestMLP, yTrainMLP, yTestMLP = train_test_split(x, y, test_size = 0.3)

# try:
#     modelMLP = MLPClassifier(hidden_layer_sizes = (10, 6)
#                              , solver = "lbfgs"
#                              , alpha = 0.001
#                              , random_state = 1
#                              , max_iter = 30000).fit(xTrainMLP, yTrainMLP)
# except Exception as e:
#     logger.exception("{}".format(e))
#     sys.exit()

# scoreMLP = modelMLP.score(xTestMLP, yTestMLP)
# print("*** scoreMLP: ", scoreMLP, " ***")

# endregion
# ----------------------神经网络算法：多层感知器----------------------#

# ----------------------极致梯度提升决策树：XGBoost----------------------#
# 注意，对于二分类问题，即objective = "binary:logistic"，则参数num_class不用设置，xgboost内部
# 会调用np.unique(np.asarray(y))判断输入的y的取值类型数量，然后对num_class设置相应的值
# region

# xTrainXGB, xTestXGB, yTrainXGB, yTestXGB = train_test_split(x, y, test_size = 0.3)

# try:
#     modelXGB = XGBClassifier(learning_rate = 0.01
#                              , n_estimators = 800
#                              , max_depth = 8
#                              , min_child_weight = 1
#                              , gamma = 0
#                              , subsample = 0.3
#                              , colsample_bytree = 0.8
#                              , colsample_bylevel = 0.7
#                              , objective = "binary:logistic"
#                              , seed = 3)
# except Exception as e:
#     logger.exception("{}".format(e))
#     sys.exit()

# modelXGB.fit(xTrainXGB, yTrainXGB)
# scoreXGB = modelXGB.score(xTestXGB, yTestXGB)
# print("*** scoreXGB: ", scoreXGB, "***")

# endregion
# ----------------------极致梯度提升决策树：XGBoost----------------------#

# ----------------------自适应增强决策树：AdaBoost----------------------#
# region

# xTrainADA, xTestADA, yTrainADA, yTestADA = train_test_split(x, y, test_size = 0.3)

# try:
#     modelADA = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2, min_samples_split = 20, min_samples_leaf = 5)
#                                   , algorithm = "SAMME"  # SAMME or SAMME.R
#                                   , n_estimators = 1000
#                                   , learning_rate = 0.8)
#     modelADA.fit(xTrainADA, yTrainADA)
# except Exception as e:
#     logger.exception("{}".format(e))
#     sys.exit()

# scoreADA = modelADA.score(xTestADA, yTestADA)
# print("*** scoreADA: " ,scoreADA, " ***")

# endregion
# ----------------------自适应增强决策树：AdaBoost----------------------#

# ----------------------画图：特征相关系数热度图----------------------#
# region

# df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mtcars.csv")
# print("*** corr ***\n", df.corr())  #计算相关系数，默认是pearson相关系数，另外可以计算kendall，spearman相关系数

# plt.figure(figsize = (12, 10), dpi = 80)
# sns.heatmap(df.corr(), xticklabels = df.corr().columns, yticklabels = df.corr().columns, cmap = "RdYlGn", center = 0, annot = True)

# plt.title("Correlogram of mtcars", fontsize = 22)
# plt.xticks(fontsize = 12)
# plt.yticks(fontsize = 12)
# plt.show()

# endregion
# ----------------------画图：特征相关系数热度图----------------------#

# ----------------------准确度、灵敏度、特异度、混淆矩阵----------------------#
# 准确度：把0预测成0的个数，加上把1预测成1的个数，与总数相比
# 灵敏度：把1预测成1的个数，与实际1的个数之比
# 特异度：把0预测成0的个数，与实际0的个数之比
# 混淆矩阵：
# 0预测成0的个数 | 0预测成1的个数
# —————————————————————————————
# 1预测成0的个数 | 1预测成1的个数
# region

# yPred = [0, 1, 0, 1, 0, 0, 1]
# yTrue = [0, 0, 0, 1, 1, 0, 1]
# print(classification_report(yTrue, yPred))
# print(confusion_matrix(yTrue, yPred))

# endregion
# ----------------------准确度、灵敏度、特异度、混淆矩阵----------------------#

# ----------------------解决数据不平衡问题：SMOTE数据合成----------------------#
# 二分类问题中，如果两类数据数量差距很大，除了可以把多的数据减少一部分外，还可以通过SMOTE合成数据
# region

# x, y = make_classification(n_classes = 2
#                            , class_sep = 2
#                            , weights = [0.9, 0.1]
#                            , n_informative = 2
#                            , n_redundant = 0
#                            , flip_y = 0
#                            , n_features = 2
#                            , n_clusters_per_class = 1
#                            , n_samples = 100
#                            , random_state = 1)
# print(Counter(y))

# plt.figure()
# sns.scatterplot(x[:,0], x[:,1], hue = y)
# plt.show()

# smo = SMOTE(random_state = 42)
# xSmo, ySmo = smo.fit_resample(x, y)
# print(Counter(ySmo))

# plt.figure()
# sns.scatterplot(xSmo[:,0], xSmo[:,1], hue = ySmo, palette = "Accent")
# sns.scatterplot(x[:,0], x[:,1], hue = y)
# plt.show()

# endregion
# ----------------------解决数据不平衡问题：SMOTE数据合成----------------------#

# ------------------------------------
dataset_path = "./data/CHAOS/CHAOS_Train/Train_Sets/CT/"
excel_lc_path = './test.csv'
paraPath = "yaml/Params.yaml"

zoom_prixl = 5  # 缩小的像素个数

# predict


def predict_features(image, mask, option_yaml_path):
    extractor = RadiomicsFeatureExtractor(option_yaml_path)
    return extractor.execute(image, mask)


# 特征抽取器是一个封装的类，用于计算影像组学特征。大量设置可用于个性化特征抽取，
# 包括：需要抽取的特征类别及其对应特征；需要使用的图像类别（原始图像/或衍生图像）；需要进行什么样的预处理。
# 我们可以使用该类的execute()方法来执行特征抽取操作。execute接受的参数为原始图像及其对应的Mask。


# -----------------------
# path=phase_path
# transform dcm and mask_png to image main function
def prepare_images(path):
    # prepare dcm

    imagePath = path + "DICOM_anon" + "/"
    maskPath = path + "Ground" + "/"

    #读取dcm格式图像
    reader = sitk.ImageSeriesReader()
    # 文件夹中包含多个series
    # 根据文件夹获取序列ID,一个文件夹里面通常是一个病人的所有切片，会分为好几个序列
    seriesIDs = reader.GetGDCMSeriesIDs(imagePath)

    # 选取其中一个序列ID,获得该序列的若干文件名
    dicom_names = reader.GetGDCMSeriesFileNames(imagePath, seriesIDs[0])
    # 设置文件名
    reader.SetFileNames(dicom_names)
    #读取dcm序列
    images_dcm = reader.Execute()
    print(images_dcm.GetSize())

    # prepare mask
    tmp = []
    for item in sorted(os.listdir(maskPath)):  #用于返回指定的文件夹包含的文件或文件夹的名字的列表
        if item.lower().endswith('.png'):  #lower变成小写
            tmp_array = sitk.GetArrayFromImage(sitk.ReadImage(maskPath +
                                                              item))  #获取图像数组
            #print(tmp_array)
            #将灰度图tmp_array中灰度值小于127的点置0，灰度值大于127的点置255
            ret, thresh = cv2.threshold(tmp_array, 127, 255, 0)  #图像阈值处理
            # 参数(1图片源，2阈值（起始值），3最大值，4表示的是这里划分的时候使用的是什么类型的算法，常用值为0)
            #thresh 二值化后的灰度图

            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )  # 得到轮廓信息 contours 轮廓本身  hierarchy 每条轮廓对应的属性
            imgnew = cv2.drawContours(tmp_array, contours, -1, (0, 0, 0),
                                      int(zoom_prixl * 2))  # 把所有轮廓画出来
            imgnew = imgnew / 255
            tmp.append(imgnew.astype(np.int32))  #数据格式转换

    images_mask = sitk.GetImageFromArray(np.array(tmp))
    print(images_mask.GetSize())
    images_mask.CopyInformation(images_dcm)

    return images_dcm, images_mask


results = list()
indexs = list()
if __name__ == '__main__':

    for patients_id in os.listdir(dataset_path):
        patients_path = dataset_path + patients_id + '/'
        # for phase in os.listdir(patients_path):
        #     phase_path = patients_path + phase + '/'
        indexs.append(str(patients_id))
        images, masks = prepare_images(patients_path)
        results.append(predict_features(images, masks, paraPath))

    df = pd.DataFrame(results)
    df.index = indexs
    df.to_csv('5_1_ct_patient.csv')