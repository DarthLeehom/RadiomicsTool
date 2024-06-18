import ui
import re
import utils
import ui
import numpy as np
import pandas as pd
import pymrmr
from log import logger
from scipy.stats import ttest_ind, levene
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    mutual_info_classif,
    chi2,
    SelectKBest,
    SelectPercentile,
)


def showTTestParamWindow():
    window = ui.createTTestParamWindow()

    while True:
        event, values = window.read(timeout=500)
        if event in (None, "Exit", "cancel"):
            break
        if event == "ok":
            if re.match("^(0|0\.\d+|1)$", values["proportiontocut"]) is None:
                ui.popupError(
                    "proportiontocut error, please check your input!")
            elif values["axis"] != "" and not values["axis"].isdigit():
                ui.popupError("axis error, please check your input!")
            elif values["permutations"] != "" and not values[
                    "permutations"].isdigit():
                ui.popupError("permutations error, please check your input!")
            elif values["randomState"] != "" and not values[
                    "randomState"].isdigit():
                ui.popupError("randomState error, please check your input!")
            elif (re.match("^(0|0\.\d+|1)$", values["trim"]) is None
                  or float(values["trim"]) >= 0.5):
                ui.popupError("trim error, please check your input!")
            else:
                utils.ttestParam["leveneCenter"] = values["center"]
                utils.ttestParam["leveneProportiontocut"] = float(
                    values["proportiontocut"])
                if values["axis"] == "":
                    utils.ttestParam["ttestAxis"] = None
                else:
                    utils.ttestParam["ttestAxis"] = int(values["axis"])
                utils.ttestParam["ttestNanPolicy"] = values["nanPolicy"]
                if values["permutations"] == "":
                    utils.ttestParam["ttestPermutations"] = None
                else:
                    utils.ttestParam["ttestPermutations"] = int(
                        values["permutations"])
                if values["randomState"] == "":
                    utils.ttestParam["ttestRandomState"] = None
                else:
                    utils.ttestParam["ttestRandomState"] = int(
                        values["randomState"])
                utils.ttestParam["ttestAlternative"] = values["alternative"]
                utils.ttestParam["ttestTrim"] = float(values["trim"])
                break
        if event == "default":
            window["center"].update("median")
            window["proportiontocut"].update("0.05")
            window["axis"].update("0")
            window["nanPolicy"].update("propagate")
            window["permutations"].update("")
            window["randomState"].update("")
            window["alternative"].update("two-sided")
            window["trim"].update("0")
    window.Close()


def showLassoParamWindow():
    window = ui.createLassoParamWindow()

    while True:
        event, values = window.read(timeout=500)
        if event in (None, "Exit", "cancel"):
            break
        if event == "ok":
            alphasGood = False
            if values["customAlphas"]:
                if (values["start"] == ""
                        or re.match("^(-?\d+)$", values["start"]) is None):
                    ui.popupError("start error, please check your input!")
                elif (values["stop"] == ""
                      or re.match("^(-?\d+)$", values["stop"]) is None):
                    ui.popupError("stop error, please check your input!")
                elif int(values["stop"]) < int(values["start"]):
                    ui.popupError("Start must greater than stop!")
                elif not values["num"].isdigit():
                    ui.popupError("num error, please check your input!")
                elif not values["base"].isdigit():
                    ui.popupError("base error, please check your input!")
                else:
                    alphasGood = True
            else:
                alphasGood = True

            if alphasGood:
                if (values["eps"] == ""
                        or re.match("^(0|0\.\d+|1)$", values["eps"]) is None):
                    ui.popupError("eps error, please check your input!")
                elif not values["nAlphas"].isdigit():
                    ui.popupError("nAlphas error, please check your input!")
                elif not values["maxIter"].isdigit():
                    ui.popupError("maxIter error, please check your input!")
                elif values["cv"] != "" and not values["cv"].isdigit():
                    ui.popupError("cv error, please check your input!")
                elif (values["randomState"] != ""
                      and not values["randomState"].isdigit()):
                    ui.popupError(
                        "randomState error, please check your input!")
                else:
                    utils.lassoParam["customAlphas"] = bool(
                        values["customAlphas"])
                    if values["customAlphas"]:
                        utils.lassoParam["start"] = int(values["start"])
                        utils.lassoParam["stop"] = int(values["stop"])
                        utils.lassoParam["num"] = int(values["num"])
                        utils.lassoParam["base"] = int(values["base"])
                    else:
                        utils.lassoParam["start"] = None
                        utils.lassoParam["stop"] = None
                        utils.lassoParam["num"] = None
                        utils.lassoParam["base"] = None
                    utils.lassoParam["eps"] = float(values["eps"])
                    utils.lassoParam["nAlphas"] = int(values["nAlphas"])
                    utils.lassoParam["fitIntercept"] = values["fitIntercept"]
                    utils.lassoParam["maxIter"] = int(values["maxIter"])
                    if values["cv"] == "":
                        utils.lassoParam["cv"] = None
                    else:
                        utils.lassoParam["cv"] = int(values["cv"])
                    if values["randomState"] == "":
                        utils.lassoParam["randomState"] = None
                    else:
                        utils.lassoParam["randomState"] = int(
                            values["randomState"])
                    utils.lassoParam["selection"] = values["selection"]
                    break
        if event == "default":
            window["customAlphas"].update(False)
            window["start"].update("")
            window["stop"].update("")
            window["num"].update("")
            window["base"].update("")
            window["eps"].update("0.001")
            window["nAlphas"].update("100")
            window["maxIter"].update("1000")
            window["cv"].update("")
            window["randomState"].update("")
            window["selection"].update("cyclic")
    window.Close()


def showPCAParamWindow():
    window = ui.createPCAParamWindow()
    while True:
        event, values = window.read(timeout=500)
        if event in (None, "Exit", "cancel"):
            break
        if event == "ok":
            if (values["nComponents"] != "mle"
                    and re.match("^(0\.\d+)$", values["nComponents"]) is None
                    and (not values["nComponents"].isdigit()
                         or int(values["nComponents"]) == 0)):
                ui.popupError("nComponents error, please check your input!")
            elif (values["iteratedPower"] != "auto"
                  and not values["iteratedPower"].isdigit()):
                ui.popupError("iteratedPower error, please check your input!")
            elif (not values["nOversamples"].isdigit()
                  or int(values["nOversamples"]) == 0):
                ui.popupError("nOversamples error, please check your input!")
            elif values["randomState"] != "" and not values[
                    "randomState"].isdigit():
                ui.popupError("randomState error, please check your input!")
            else:
                if values["nComponents"] == "":
                    utils.pcaParam["nComponents"] = None
                elif values["nComponents"] == "mle":
                    utils.pcaParam["nComponents"] = values["nComponents"]
                elif values["nComponents"].isdigit():
                    utils.pcaParam["nComponents"] = int(values["nComponents"])
                else:
                    utils.pcaParam["nComponents"] = float(
                        values["nComponents"])
                utils.pcaParam["whiten"] = bool(values["whiten"])
                utils.pcaParam["svdSolver"] = values["svdSolver"]
                if values["iteratedPower"].isdigit():
                    utils.pcaParam["iteratedPower"] = int(
                        values["iteratedPower"])
                else:
                    utils.pcaParam["iteratedPower"] = values["iteratedPower"]
                utils.pcaParam["nOversamples"] = int(values["nOversamples"])
                utils.pcaParam["powerIterationNormalizer"] = values[
                    "powerIterationNormalizer"]
                if values["randomState"] == "":
                    utils.pcaParam["randomState"] = None
                else:
                    utils.pcaParam["randomState"] = int(values["randomState"])
                break
        if event == "default":
            window["nComponents"].update("")
            window["whiten"].update(False)
            window["svdSolver"].update("auto")
            window["iteratedPower"].update("auto")
            window["nOversamples"].update(10)
            window["powerIterationNormalizer"].update("auto")
            window["randomState"].update("")
    window.Close()


def showMUIFParamWindow():
    window = ui.createMUIFParamWindow()
    while True:
        event, values = window.read(timeout=500)
        if event in (None, "Exit", "cancel"):
            break
        if event == "ok":
            if (values["useKBest"] and values["k"] != "all"
                    and (not values["k"].isdigit() or int(values["k"]) == 0)):
                ui.popupError("k error, please check your input!")
            elif values["usePercentile"] and (
                    not values["percentile"].isdigit() or
                (int(values["percentile"]) <= 0
                 or int(values["percentile"]) > 100)):
                ui.popupError("percentile error, please check your input!")
            else:
                utils.muifParam["useKBest"] = bool(values["useKBest"])
                if utils.muifParam["useKBest"]:
                    if values["k"] == "auto":
                        utils.muifParam["k"] = values["k"]
                    else:
                        utils.muifParam["k"] = int(values["k"])
                else:
                    utils.muifParam["usePercentile"] = bool(
                        values["usePercentile"])
                    utils.muifParam["percentile"] = int(values["percentile"])
                break
        if event == "default":
            window["useKBest"].update(True)
            window["k"].update(10)
            window["usePercentile"].update(False)
            window["percentile"].update(10)
    window.Close()


def showChi2ParamWindow():
    window = ui.createChi2ParamWindow()
    while True:
        event, values = window.read(timeout=500)
        if event in (None, "Exit", "cancel"):
            break
        if event == "ok":
            if (values["useKBest"] and values["k"] != "all"
                    and (not values["k"].isdigit() or int(values["k"]) == 0)):
                ui.popupError("k error, please check your input!")
            elif values["usePercentile"] and (
                    not values["percentile"].isdigit() or
                (int(values["percentile"]) <= 0
                 or int(values["percentile"]) > 100)):
                ui.popupError("percentile error, please check your input!")
            else:
                utils.chi2Param["useKBest"] = bool(values["useKBest"])
                if utils.chi2Param["useKBest"]:
                    if values["k"] == "auto":
                        utils.chi2Param["k"] = values["k"]
                    else:
                        utils.chi2Param["k"] = int(values["k"])
                else:
                    utils.chi2Param["usePercentile"] = bool(
                        values["usePercentile"])
                    utils.chi2Param["percentile"] = int(values["percentile"])
                break
        if event == "default":
            window["useKBest"].update(True)
            window["k"].update(10)
            window["usePercentile"].update(False)
            window["percentile"].update(10)
    window.Close()


def showmRMRParamWindow():
    window = ui.createmRMRParamWindow()
    while True:
        event, values = window.read(timeout=500)
        if event in (None, "Exit", "cancel"):
            break
        if event == "ok":
            if not values["nfeats"].isdigit() or int(values["nfeats"]) == 0:
                ui.popupError("nfeats error, please check your input!")
            else:
                utils.mRMRParam["method"] = values["method"]
                utils.mRMRParam["nfeats"] = int(values["nfeats"])
                break
        if event == "default":
            window["method"].update("MIQ")
            window["nfeats"].update(8)
    window.Close()


def runTTest(x, pData, nData, msgCallBack):
    indexTTest = []
    for colName in x.columns:
        try:
            if (levene(
                    pData[colName],
                    nData[colName],
                    center=utils.ttestParam["leveneCenter"],
                    proportiontocut=utils.ttestParam["leveneProportiontocut"],
            )[1] > 0.05):
                if (ttest_ind(
                        pData[colName],
                        nData[colName],
                        equal_var=True,
                        axis=utils.ttestParam["ttestAxis"],
                        nan_policy=utils.ttestParam["ttestNanPolicy"],
                        permutations=utils.ttestParam["ttestPermutations"],
                        random_state=utils.ttestParam["ttestRandomState"],
                        alternative=utils.ttestParam["ttestAlternative"],
                        trim=utils.ttestParam["ttestTrim"],
                )[1] < 0.05):
                    indexTTest.append(colName)
            else:
                if (ttest_ind(
                        pData[colName],
                        nData[colName],
                        equal_var=False,
                        axis=utils.ttestParam["ttestAxis"],
                        nan_policy=utils.ttestParam["ttestNanPolicy"],
                        permutations=utils.ttestParam["ttestPermutations"],
                        random_state=utils.ttestParam["ttestRandomState"],
                        alternative=utils.ttestParam["ttestAlternative"],
                        trim=utils.ttestParam["ttestTrim"],
                )[1] < 0.05):
                    indexTTest.append(colName)
        except Exception as e:
            logger.exception("{}, {}".format(colName, e))
            ui.popupError("{}, {}".format(colName, e))
            return False, None

    msgCallBack("*** features num after ttest: {} ***".format(len(indexTTest)))

    x = x[indexTTest]
    x = x.apply(pd.to_numeric, errors="ignore")

    msgCallBack("*** x head ***\n {}".format(x.head()))
    return True, x


def runLasso(x, y, msgCallBack):
    msgCallBack("*** x shape: {} ***".format(x.shape))
    if utils.lassoParam["customAlphas"]:
        alphas = np.logspace(
            start=utils.lassoParam["start"],
            stop=utils.lassoParam["stop"],
            num=utils.lassoParam["num"],
            base=utils.lassoParam["base"],
        )
    else:
        alphas = None

    try:
        modelLassoCV = LassoCV(
            alphas=alphas,
            eps=utils.lassoParam["eps"],
            n_alphas=utils.lassoParam["nAlphas"],
            fit_intercept=utils.lassoParam["fitIntercept"],
            max_iter=utils.lassoParam["maxIter"],
            cv=utils.lassoParam["cv"],
            random_state=utils.lassoParam["randomState"],
            selection=utils.lassoParam["selection"],
        ).fit(x, y.values.ravel())
    except Exception as e:
        logger.exception("{}".format(e))
        ui.popupError("{}".format(e))
        return False, None, None

    msgCallBack("*** Lasso lambda: {} ***".format(modelLassoCV.alpha_))
    coef = pd.Series(modelLassoCV.coef_, index=x.columns)
    msgCallBack(
        "*** Lasso picked {} variables and eliminated the other {} ***".format(
            sum(coef != 0), sum(coef == 0)))
    indexLasso = coef[coef != 0].index
    x = x[indexLasso]
    msgCallBack("*** coef != 0 ***\n {}".format(coef[coef != 0]))
    return True, x, modelLassoCV


def runPCA(x, y, msgCallBack):
    # 特征降维和特征筛选的区别：
    # 特征降维是找到特征之间的映射关系，通过映射关系将多个特征合并，映射后特征值会发生变化；
    # 特征筛选就是单纯的从原始特征中选择部分特征，特征选择前后的值不变
    msgCallBack("*** x shape: {} ***".format(x.shape))
    try:
        modelPCA = PCA(
            n_components=utils.pcaParam["nComponents"],
            whiten=utils.pcaParam["whiten"],
            svd_solver=utils.pcaParam["svdSolver"],
            iterated_power=utils.pcaParam["iteratedPower"],
            n_oversamples=utils.pcaParam["nOversamples"],
            power_iteration_normalizer=utils.
            pcaParam["powerIterationNormalizer"],
            random_state=utils.pcaParam["randomState"],
        )  # 标识降维后包含原来n_components%的特征
        modelPCA.fit(x, y.values.ravel())
        x = pd.DataFrame(modelPCA.transform(x))
    except Exception as e:
        logger.exception("{}".format(e))
        ui.popupError("{}".format(e))
        return False, None, None
    msgCallBack("*** PCA variance ratio ***\n {}".format(
        modelPCA.explained_variance_ratio_))
    msgCallBack("*** x shape: {} ***".format(x.shape))
    return True, x, modelPCA


def runMUIF(x, y, msgCallBack):
    # 值越大，表示相关性越高
    # 这里选择特征可以使用sklearn.feature_selection提供的SelectKBest和SelectPercentile
    # SelectKBest表示选择K个相关性最高的特征
    # SelectPercentile表示按照相关性从高到低，选择多少百分比的特征

    try:
        if utils.muifParam["useKBest"]:
            x = SelectKBest(mutual_info_classif,
                            k=utils.muifParam["k"]).fit_transform(
                                x, y.values.ravel())
            msgCallBack("*** x shape after MUIF on SelectKBest: {} ***".format(
                x.shape))
        else:
            x = SelectPercentile(
                mutual_info_classif,
                percentile=utils.muifParam["percentile"]).fit_transform(
                    x, y.values.ravel())
            msgCallBack(
                "*** x shape after MUIF on SelectPercentile: {} ***".format(
                    x.shape))
    except Exception as e:
        logger.exception("{}".format(e))
        ui.popupError("{}".format(e))
        return False, None
    return True, x


def runChi2(x, y, msgCallBack):
    # 值越大，表示相关性越高
    # 这里选择特征可以使用sklearn.feature_selection提供的SelectKBest和SelectPercentile
    # SelectKBest表示选择K个相关性最高的特征
    # SelectPercentile表示按照相关性从高到低，选择多少百分比的特征

    try:
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)  # 卡方检测x不能为负数，所以先归一化到[0, 1]
        if utils.chi2Param["useKBest"]:
            x = SelectKBest(chi2, k=utils.chi2Param["k"]).fit_transform(
                x, y.values.ravel())
            msgCallBack("*** x shape after Chi2 on SelectKBest: {} ***".format(
                x.shape))
        else:
            x = SelectPercentile(
                chi2, percentile=utils.chi2Param["percentile"]).fit_transform(
                    x, y.values.ravel())
            msgCallBack(
                "*** x shape after Chi2 on SelectPercentile: {} ***".format(
                    x.shape))
    except Exception as e:
        logger.exception("{}".format(e))
        ui.popupError("{}".format(e))
        return False, None
    return True, x


def runmRMR(x, y, msgCallBack):
    # 核心思想是从给定的特征集合中寻找与目标量有最大相关性且特征相互之间具有最少冗余性的特征子集。
    # 函数一共三个参数：
    # 参数一：DataFrame，要求第一列是目标量，其他列是特征量，其中首行必须是特征名称，且必须是字符形式，例如a1,a2，等
    # 参数二：选择的方法，有'MID'、'MIQ'两种。MID是基于互信息的mRMR，MIQ是基于商的mRMR。
    # 参数三：要求int类型，最后输出的特征数量。

    datamRMR = pd.concat([y, x], axis=1)
    msgCallBack("*** dataMRMR head ***\n {}".format(datamRMR.head()))

    try:
        mrmr = pymrmr.mRMR(datamRMR,
                           method=utils.mRMRParam["method"],
                           nfeats=utils.mRMRParam["nfeats"])
    except Exception as e:
        logger.exception("{}".format(e))
        ui.popupError("{}".format(e))
        return False, None

    msgCallBack("*** mrmr ***\n {}".format(mrmr))

    x = datamRMR[mrmr].copy()
    msgCallBack("*** x after mRMR ***\n {} {}".format(x.shape, x.head()))
    return True, x


showFeatureSelectionParamWindow = {
    "ttestParam": showTTestParamWindow,
    "lassoParam": showLassoParamWindow,
    "pcaParam": showPCAParamWindow,
    "muifParam": showMUIFParamWindow,
    "chi2Param": showChi2ParamWindow,
    "mRMRParam": showmRMRParamWindow,
}
