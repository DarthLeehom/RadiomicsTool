import ui
import re
import utils
import numpy as np
from log import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def showRandomForestParamWindow():
    window = ui.createRandomForestParamWindow()

    while True:
        event, values = window.read(timeout=500)
        if event in (None, "Exit", "cancel"):
            break
        if event == "ok":
            if not values["nEstimators"].isdigit() or int(
                    values["nEstimators"]) == 0:
                ui.popupError("nEstimators error, please check your input!")
            elif values["maxDepth"] != "" and (
                    not values["maxDepth"].isdigit()
                    or int(values["maxDepth"]) == 0):
                ui.popupError("maxDepth error, please check your input!")
            elif (not values["minSamplesSplit"].isdigit()
                  or int(values["minSamplesSplit"]) == 0) and re.match(
                      "^(0\.\d+|1)$", values["minSamplesSplit"]) is None:
                ui.popupError(
                    "minSamplesSplit error, please check your input!")
            elif (not values["minSamplesLeaf"].isdigit()
                  or int(values["minSamplesLeaf"]) == 0) and re.match(
                      "^(0\.\d+|1)$", values["minSamplesLeaf"]) is None:
                ui.popupError("minSamplesLeaf error, please check your input!")
            elif re.match("^(0|0\.\d+|1)$",
                          values["minWeightFractionLeaf"]) is None:
                ui.popupError(
                    "minWeightFractionLeaf error, please check your input!")
            elif (
                    values["maxFeatures"] != ""
                    and values["maxFeatures"] != "sqrt"
                    and values["maxFeatures"] != "log2" and
                ((not values["maxFeatures"].isdigit()
                  or int(values["maxFeatures"]) == 0)
                 and re.match("^(0\.\d+|1)$", values["maxFeatures"]) is None)):
                ui.popupError("maxFeatures error, please check your input!")
            elif values["maxLeafNodes"] != "" and (
                    not values["maxLeafNodes"].isdigit()
                    or int(values["maxLeafNodes"]) == 0):
                ui.popupError("maxLeafNodes error, please check your input!")
            elif (re.match(
                    "^[1-9]\d*(\.\d+)?|0\.\d*[1-9]\d*|0$",
                    values["minImpurityDecrease"],
            ) is None):
                ui.popupError(
                    "minImpurityDecrease error, please check your input!")
            elif values["randomState"] != "" and not values[
                    "randomState"].isdigit():
                ui.popupError("randomState error, please check your input!")
            elif (re.match(
                    "^[1-9]\d*\.\d+|0\.\d*[1-9]\d*|0$",
                    values["ccpAlpha"],
            ) is None):
                ui.popupError("ccpAlpha error, please check your input!")
            elif values["maxSamples"] != "" and (
                (not values["maxSamples"].isdigit()
                 or int(values["maxSamples"]) == 0) and re.match(
                     "^(0\.\d+|1)$", values["maxSamples"]) is None):
                ui.popupError("maxSamples error, please check your input!")
            else:
                utils.randomForestParam["nEstimators"] = int(
                    values["nEstimators"])
                utils.randomForestParam["criterion"] = values["criterion"]
                if values["maxDepth"] == "":
                    utils.randomForestParam["maxDepth"] = None
                else:
                    utils.randomForestParam["maxDepth"] = int(
                        values["maxDepth"])
                if values["minSamplesSplit"].isdigit():
                    utils.randomForestParam["minSamplesSplit"] = int(
                        values["minSamplesSplit"])
                else:
                    utils.randomForestParam["minSamplesSplit"] = float(
                        values["minSamplesSplit"])
                if values["minSamplesLeaf"].isdigit():
                    utils.randomForestParam["minSamplesLeaf"] = int(
                        values["minSamplesLeaf"])
                else:
                    utils.randomForestParam["minSamplesLeaf"] = float(
                        values["minSamplesLeaf"])
                utils.randomForestParam["minWeightFractionLeaf"] = float(
                    values["minWeightFractionLeaf"])
                if values["maxFeatures"] == "":
                    utils.randomForestParam["maxFeatures"] = None
                elif values["maxFeatures"] == "sqrt":
                    utils.randomForestParam["maxFeatures"] = "sqrt"
                elif values["maxFeatures"] == "log2":
                    utils.randomForestParam["maxFeatures"] = "log2"
                elif values["maxFeatures"].isdigit():
                    utils.randomForestParam["maxFeatures"] = int(
                        values["maxFeatures"])
                else:
                    utils.randomForestParam["maxFeatures"] = float(
                        values["maxFeatures"])
                if values["maxLeafNodes"] == "":
                    utils.randomForestParam["maxLeafNodes"] = None
                else:
                    utils.randomForestParam["maxLeafNodes"] = int(
                        values["maxLeafNodes"])
                utils.randomForestParam["minImpurityDecrease"] = float(
                    values["minImpurityDecrease"])
                utils.randomForestParam["bootstrap"] = values["bootstrap"]
                utils.randomForestParam["oobScore"] = values["oobScore"]
                if values["randomState"] == "":
                    utils.randomForestParam["randomState"] = None
                else:
                    utils.randomForestParam["randomState"] = int(
                        values["randomState"])
                utils.randomForestParam["warmStart"] = values["warmStart"]
                if values["classWeight"] == "":
                    utils.randomForestParam["classWeight"] = None
                else:
                    utils.randomForestParam["classWeight"] = values[
                        "classWeight"]
                utils.randomForestParam["ccpAlpha"] = float(values["ccpAlpha"])
                if values["maxSamples"] == "":
                    utils.randomForestParam["maxSamples"] = None
                elif values["maxSamples"].isdigit():
                    utils.randomForestParam["maxSamples"] = int(
                        values["maxSamples"])
                else:
                    utils.randomForestParam["maxSamples"] = float(
                        values["maxSamples"])
                break
        if event == "default":
            window["nEstimators"].update(100)
            window["criterion"].update("gini")
            window["maxDepth"].update("")
            window["minSamplesSplit"].update(2)
            window["minSamplesLeaf"].update(1)
            window["minWeightFractionLeaf"].update(0)
            window["maxFeatures"].update("sqrt")
            window["maxLeafNodes"].update("")
            window["minImpurityDecrease"].update(0)
            window["bootstrap"].update(True)
            window["oobScore"].update(False)
            window["randomState"].update("")
            window["warmStart"].update(False)
            window["classWeight"].update("")
            window["ccpAlpha"].update(0)
            window["maxSamples"].update("")
    window.Close()


def showSVMParamWindow():
    window = ui.createSVMParamWindow()

    while True:
        event, values = window.read(timeout=500)
        if event in (None, "Exit", "cancel"):
            break
        if event == "ok":
            if (re.match("^[1-9]\d*(\.\d+)?|0\.\d*[1-9]\d*$", values["C"])
                    is None or float(values["C"]) == 0):
                ui.popupError("C error, please check your input!")
            elif not values["degree"].isdigit():
                ui.popupError("degree error, please check your input!")
            elif (values["gamma"] != "scale" and values["gamma"] != "auto"
                  and re.match("^[1-9]\d*(\.\d+)?|0\.\d*[1-9]\d*|0$",
                               values["gamma"]) is None):
                ui.popupError("gamma error, please check your input!")
            elif (re.match("^[1-9]\d*(\.\d+)?|0\.\d*[1-9]\d*|0$",
                           values["coef0"]) is None):
                ui.popupError("coef0 error, please check your input!")
            elif values["maxIter"] != "-1" and (
                    not values["maxIter"].isdigit()
                    or int(values["maxIter"] == 0)):
                ui.popupError("maxIter error, please check your input!")
            elif values["randomState"] != "" and not values[
                    "randomState"].isdigit():
                ui.popupError("randomState error, please check your input!")
            else:
                utils.svmParam["C"] = float(values["C"])
                utils.svmParam["kernel"] = values["kernel"]
                utils.svmParam["degree"] = int(values["degree"])
                if values["gamma"] == "scale" or values["gamma"] == "auto":
                    utils.svmParam["gamma"] = values["gamma"]
                else:
                    utils.svmParam["gamma"] = float(values["gamma"])
                utils.svmParam["coef0"] = float(values["coef0"])
                utils.svmParam["shrinking"] = values["shrinking"]
                utils.svmParam["probability"] = values["probability"]
                if values["classWeight"] == "":
                    utils.svmParam["classWeight"] = None
                else:
                    utils.svmParam["classWeight"] = values["classWeight"]
                if values["maxIter"] == "-1":
                    utils.svmParam["maxIter"] = -1
                else:
                    utils.svmParam["maxIter"] = int(values["maxIter"])
                utils.svmParam["decisionFunctionShape"] = values[
                    "decisionFunctionShape"]
                utils.svmParam["breakTies"] = values["breakTies"]
                if values["randomState"] == "":
                    utils.svmParam["randomState"] = None
                else:
                    utils.svmParam["randomState"] = int(values["randomState"])
                break
        if event == "default":
            window["C"].update(1)
            window["kernel"].update("rbf")
            window["degree"].update(3)
            window["gamma"].update("scale")
            window["coef0"].update("0")
            window["shrinking"].update(True)
            window["probability"].update(False)
            window["classWeight"].update("")
            window["maxIter"].update("-1")
            window["decisionFunctionShape"].update("ovr")
            window["breakTies"].update(False)
            window["randomState"].update("")
    window.Close()


def showMLPParamWindow():
    window = ui.createMLPParamWindow()

    while True:
        event, values = window.read(timeout=500)
        if event in (None, "Exit", "cancel"):
            break
        if event == "ok":
            hiddenLayerSizesError = False
            for str in values["hiddenLayerSizes"].split(","):
                if not str.isdigit() or int(str) == 0:
                    ui.popupError(
                        "hiddenLayerSizes error, please check your input!")
                    hiddenLayerSizesError = True
                    break
            if not hiddenLayerSizesError:
                if (re.match("^[1-9]\d*(\.\d+)?|0\.\d*[1-9]\d*$",
                             values["alpha"]) is None):
                    ui.popupError("alpha error, please check your input!")
                elif values["batchSize"] != "auto" and (
                        values["batchSize"].isdigit() is None
                        or int(values["batchSize"]) == 0):
                    ui.popupError("batchSize error, please check your input!")
                elif (re.match("^[1-9]\d*(\.\d+)?|0\.\d*[1-9]\d*$",
                               values["learningRateInit"]) is None):
                    ui.popupError(
                        "learningRateInit error, please check your input!")
                elif (re.match("^[1-9]\d*(\.\d+)?|0\.\d*[1-9]\d*|0$",
                               values["powerT"]) is None):
                    ui.popupError("powerT error, please check your input!")
                elif not values["maxIter"].isdigit() or int(
                        values["maxIter"]) == 0:
                    ui.popupError("maxIter error, please check your input!")
                elif (values["randomState"] != ""
                      and not values["randomState"].isdigit()):
                    ui.popupError(
                        "randomState error, please check your input!")
                elif re.match("^(0\.\d+)$", values["momentum"]) is None:
                    ui.popupError("momentum error, please check your input!")
                elif re.match("^(0\.\d+)$",
                              values["validationFraction"]) is None:
                    ui.popupError(
                        "validationFraction error, please check your input!")
                elif re.match("^(0|0\.\d+)$", values["beta1"]) is None:
                    ui.popupError("beta1 error, please check your input!")
                elif re.match("^(0|0\.\d+)$", values["beta2"]) is None:
                    ui.popupError("beta2 error, please check your input!")
                elif (not values["nIterNoChange"].isdigit()
                      or int(values["nIterNoChange"]) == 0):
                    ui.popupError(
                        "nIterNoChange error, please check your input!")
                elif not values["maxFun"].isdigit() or int(
                        values["maxFun"]) == 0:
                    ui.popupError("maxFun error, please check your input!")
                else:
                    hiddenLayerSizesList = values["hiddenLayerSizes"].split(
                        ",")
                    utils.mlpParam["hiddenLayerSizes"] = tuple(
                        map(int, hiddenLayerSizesList))
                    utils.mlpParam["activation"] = values["activation"]
                    utils.mlpParam["solver"] = values["solver"]
                    utils.mlpParam["alpha"] = float(values["alpha"])
                    if values["batchSize"] == "auto":
                        utils.mlpParam["alpha"] = values["batchSize"]
                    else:
                        utils.mlpParam["alpha"] = int(values["batchSize"])
                    utils.mlpParam["learningRate"] = values["learningRate"]
                    utils.mlpParam["learningRateInit"] = float(
                        values["learningRateInit"])
                    utils.mlpParam["powerT"] = float(values["powerT"])
                    utils.mlpParam["maxIter"] = int(values["maxIter"])
                    utils.mlpParam["shuffle"] = values["shuffle"]
                    if values["randomState"] == "":
                        utils.mlpParam["randomState"] = None
                    else:
                        utils.mlpParam["randomState"] = int(
                            values["randomState"])
                    utils.mlpParam["warmStart"] = values["warmStart"]
                    utils.mlpParam["momentum"] = float(values["momentum"])
                    utils.mlpParam["nesterovsMomentum"] = values[
                        "nesterovsMomentum"]
                    utils.mlpParam["earlyStopping"] = values["earlyStopping"]
                    utils.mlpParam["validationFraction"] = float(
                        values["validationFraction"])
                    utils.mlpParam["beta1"] = float(values["beta1"])
                    utils.mlpParam["beta2"] = float(values["beta2"])
                    utils.mlpParam["nIterNoChange"] = int(
                        values["nIterNoChange"])
                    utils.mlpParam["maxFun"] = int(values["maxFun"])
                    break
        if event == "default":
            window["hiddenLayerSizes"].update("100")
            window["activation"].update("relu")
            window["solver"].update("adam")
            window["alpha"].update(0.0001)
            window["batchSize"].update("auto")
            window["learningRate"].update("constant")
            window["learningRateInit"].update(0.001)
            window["powerT"].update(0.5)
            window["maxIter"].update(200)
            window["shuffle"].update(True)
            window["randomState"].update("")
            window["warmStart"].update(False)
            window["momentum"].update(0.9)
            window["nesterovsMomentum"].update(True)
            window["earlyStopping"].update(False)
            window["validationFraction"].update(0.1)
            window["beta1"].update(0.9)
            window["beta2"].update(0.999)
            window["nIterNoChange"].update(10)
            window["maxFun"].update(15000)
    window.Close()


def showXGBoostParamWindow():
    window = ui.createXGBoostParamWindow()

    while True:
        event, values = window.read(timeout=500)
        if event in (None, "Exit", "cancel"):
            break
        if event == "ok":
            if values["maxDepth"] != "" and (not values["maxDepth"].isdigit()
                                             or int(values["maxDepth"]) == 0):
                ui.popupError("maxDepth error, please check your input!")
            elif values["maxLeaves"] != "" and not values["maxLeaves"].isdigit(
            ):
                ui.popupError("maxLeaves error, please check your input!")
            elif values["maxBin"] != "" and (not values["maxBin"].isdigit()
                                             or int(values["maxBin"]) == 0):
                ui.popupError("maxBin error, please check your input!")
            elif (values["learningRate"] != "" and re.match(
                    "^(0|0\.\d+|1)$", values["learningRate"]) is None):
                ui.popupError("learningRate error, please check your input!")
            elif not values["nEstimators"].isdigit() or int(
                    values["nEstimators"]) == 0:
                ui.popupError("nEstimators error, please check your input!")
            elif (values["gamma"] != "" and re.match(
                    "^[1-9]\d*(\.\d+)?|0\.\d*[1-9]\d*|0$",
                    values["gamma"],
            ) is None):
                ui.popupError("gamma error, please check your input!")
            elif (values["minChildWeight"] != "" and re.match(
                    "^[1-9]\d*(\.\d+)?|0\.\d*[1-9]\d*|0$",
                    values["minChildWeight"],
            ) is None):
                ui.popupError("minChildWeight error, please check your input!")
            elif (values["maxDeltaStep"] != "" and re.match(
                    "^[1-9]\d*(\.\d+)?|0\.\d*[1-9]\d*|0$",
                    values["maxDeltaStep"],
            ) is None):
                ui.popupError("maxDeltaStep error, please check your input!")
            elif (values["subsample"] != ""
                  and re.match("^(0\.\d+|1)$", values["subsample"]) is None):
                ui.popupError("subsample error, please check your input!")
            elif (values["colsampleBytree"] != "" and re.match(
                    "^(0\.\d+|1)$", values["colsampleBytree"]) is None):
                ui.popupError(
                    "colsampleBytree error, please check your input!")
            elif (values["colsampleBylevel"] != "" and re.match(
                    "^(0\.\d+|1)$", values["colsampleBylevel"]) is None):
                ui.popupError(
                    "colsampleBylevel error, please check your input!")
            elif (values["colsampleBynode"] != "" and re.match(
                    "^(0\.\d+|1)$", values["colsampleBynode"]) is None):
                ui.popupError(
                    "colsampleBynode error, please check your input!")
            elif values["scalePosWeight"] != "" and (
                    not values["scalePosWeight"].isdigit()
                    or int(values["scalePosWeight"]) == 0):
                ui.popupError("scalePosWeight error, please check your input!")
            elif (values["baseScore"] != ""
                  and re.match("^(0|0\.\d+|1)$", values["baseScore"]) is None):
                ui.popupError("baseScore error, please check your input!")
            elif values["randomState"] != "" and not values[
                    "randomState"].isdigit():
                ui.popupError("randomState error, please check your input!")
            else:
                if values["maxDepth"] == "":
                    utils.xgboostParam["maxDepth"] = None
                else:
                    utils.xgboostParam["maxDepth"] = int(values["maxDepth"])
                if values["maxLeaves"] == "":
                    utils.xgboostParam["maxDepth"] = None
                else:
                    utils.xgboostParam["maxLeaves"] = int(values["maxDepth"])
                if values["maxBin"] == "":
                    utils.xgboostParam["maxBin"] = None
                else:
                    utils.xgboostParam["maxBin"] = int(values["maxBin"])
                if values["growPolicy"] == "":
                    utils.xgboostParam["growPolicy"] = None
                else:
                    utils.xgboostParam["growPolicy"] = values["growPolicy"]
                if values["learningRate"] == "":
                    utils.xgboostParam["learningRate"] = None
                else:
                    utils.xgboostParam["learningRate"] = float(
                        values["learningRate"])
                utils.xgboostParam["nEstimators"] = int(values["nEstimators"])
                utils.xgboostParam["objective"] = values["objective"]
                if values["booster"] == "":
                    utils.xgboostParam["booster"] = None
                else:
                    utils.xgboostParam["booster"] = values["booster"]
                if values["treeMethod"] == "":
                    utils.xgboostParam["treeMethod"] = None
                else:
                    utils.xgboostParam["treeMethod"] = values["treeMethod"]
                if values["gamma"] == "":
                    utils.xgboostParam["gamma"] = None
                else:
                    utils.xgboostParam["gamma"] = float(values["gamma"])
                if values["minChildWeight"] == "":
                    utils.xgboostParam["minChildWeight"] = None
                else:
                    utils.xgboostParam["minChildWeight"] = float(
                        values["minChildWeight"])
                if values["maxDeltaStep"] == "":
                    utils.xgboostParam["maxDeltaStep"] = None
                else:
                    utils.xgboostParam["maxDeltaStep"] = float(
                        values["maxDeltaStep"])
                if values["subsample"] == "":
                    utils.xgboostParam["subsample"] = None
                else:
                    utils.xgboostParam["subsample"] = float(
                        values["subsample"])
                if values["samplingMethod"] == "":
                    utils.xgboostParam["samplingMethod"] = None
                else:
                    utils.xgboostParam["samplingMethod"] = values[
                        "samplingMethod"]
                if values["colsampleBytree"] == "":
                    utils.xgboostParam["colsampleBytree"] = None
                else:
                    utils.xgboostParam["colsampleBytree"] = float(
                        values["colsampleBytree"])
                if values["colsampleBylevel"] == "":
                    utils.xgboostParam["colsampleBylevel"] = None
                else:
                    utils.xgboostParam["colsampleBylevel"] = float(
                        values["colsampleBylevel"])
                if values["colsampleBynode"] == "":
                    utils.xgboostParam["colsampleBynode"] = None
                else:
                    utils.xgboostParam["colsampleBynode"] = float(
                        values["colsampleBynode"])
                if values["scalePosWeight"] == "":
                    utils.xgboostParam["scalePosWeight"] = None
                else:
                    utils.xgboostParam["scalePosWeight"] = int(
                        values["scalePosWeight"])
                if values["baseScore"] == "":
                    utils.xgboostParam["baseScore"] = None
                else:
                    utils.xgboostParam["baseScore"] = float(
                        values["baseScore"])
                if values["randomState"] == "":
                    utils.xgboostParam["randomState"] = None
                else:
                    utils.xgboostParam["randomState"] = int(
                        values["randomState"])
                break
        if event == "default":
            window["maxDepth"].update("")
            window["maxLeaves"].update("")
            window["maxBin"].update("")
            window["growPolicy"].update("")
            window["learningRate"].update("")
            window["nEstimators"].update(100)
            window["objective"].update("binary:logistic")
            window["booster"].update("")
            window["treeMethod"].update("")
            window["gamma"].update("")
            window["minChildWeight"].update("")
            window["maxDeltaStep"].update("")
            window["subsample"].update("")
            window["samplingMethod"].update("")
            window["colsampleBytree"].update("")
            window["colsampleBylevel"].update("")
            window["colsampleBynode"].update("")
            window["scalePosWeight"].update("")
            window["baseScore"].update("")
            window["randomState"].update("")
    window.Close()


def showAdaBoostParamWindow():
    window = ui.createAdaBoostParamWindow()

    while True:
        event, values = window.read(timeout=500)
        if event in (None, "Exit", "cancel"):
            break
        if event == "ok":
            if values["maxDepth"] != "" and (not values["maxDepth"].isdigit()
                                             or int(values["maxDepth"]) == 0):
                ui.popupError("maxDepth error, please check your input!")
            elif (not values["minSamplesSplit"].isdigit()
                  or int(values["minSamplesSplit"]) == 0) and re.match(
                      "^(0\.\d+|1)$", values["minSamplesSplit"]) is None:
                ui.popupError(
                    "minSamplesSplit error, please check your input!")
            elif (not values["minSamplesLeaf"].isdigit()
                  or int(values["minSamplesLeaf"]) == 0) and re.match(
                      "^(0\.\d+|1)$", values["minSamplesLeaf"]) is None:
                ui.popupError("minSamplesLeaf error, please check your input!")
            elif re.match("^(0|0\.\d+|1)$",
                          values["minWeightFractionLeaf"]) is None:
                ui.popupError(
                    "minWeightFractionLeaf error, please check your input!")
            elif (
                    values["maxFeatures"] != ""
                    and values["maxFeatures"] != "sqrt"
                    and values["maxFeatures"] != "log2" and
                ((not values["maxFeatures"].isdigit()
                  or int(values["maxFeatures"]) == 0)
                 and re.match("^(0\.\d+|1)$", values["maxFeatures"]) is None)):
                ui.popupError("maxFeatures error, please check your input!")
            elif (values["dtRandomState"] != ""
                  and not values["dtRandomState"].isdigit()):
                ui.popupError(
                    "DecisionTree randomState error, please check your input!")
            elif values["maxLeafNodes"] != "" and (
                    not values["maxLeafNodes"].isdigit()
                    or int(values["maxLeafNodes"]) == 0):
                ui.popupError("maxLeafNodes error, please check your input!")
            elif (re.match(
                    "^[1-9]\d*(\.\d+)?|0\.\d*[1-9]\d*|0$",
                    values["minImpurityDecrease"],
            ) is None):
                ui.popupError(
                    "minImpurityDecrease error, please check your input!")
            elif (re.match(
                    "^[1-9]\d*\.\d+|0\.\d*[1-9]\d*|0$",
                    values["ccpAlpha"],
            ) is None):
                ui.popupError("ccpAlpha error, please check your input!")
            elif not values["nEstimators"].isdigit() or int(
                    values["nEstimators"]) == 0:
                ui.popupError("nEstimators error, please check your input!")
            elif (re.match("^[1-9]\d*(\.\d+)?|0\.\d*[1-9]\d*$",
                           values["learningRate"]) is None):
                ui.popupError("learningRate error, please check your input!")
            elif (values["abRandomState"] != ""
                  and not values["abRandomState"].isdigit()):
                ui.popupError(
                    "Adaboost randomState error, please check your input!")
            else:
                utils.adaboostParam["criterion"] = values["criterion"]
                utils.adaboostParam["splitter"] = values["splitter"]
                if values["maxDepth"] == "":
                    utils.adaboostParam["maxDepth"] = None
                else:
                    utils.adaboostParam["maxDepth"] = int(values["maxDepth"])
                if values["minSamplesSplit"].isdigit():
                    utils.adaboostParam["minSamplesSplit"] = int(
                        values["minSamplesSplit"])
                else:
                    utils.adaboostParam["minSamplesSplit"] = float(
                        values["minSamplesSplit"])
                if values["minSamplesLeaf"].isdigit():
                    utils.adaboostParam["minSamplesLeaf"] = int(
                        values["minSamplesLeaf"])
                else:
                    utils.adaboostParam["minSamplesLeaf"] = float(
                        values["minSamplesLeaf"])
                utils.adaboostParam["minWeightFractionLeaf"] = float(
                    values["minWeightFractionLeaf"])
                if values["maxFeatures"] == "":
                    utils.adaboostParam["maxFeatures"] = None
                elif values["maxFeatures"] == "sqrt":
                    utils.adaboostParam["maxFeatures"] = "sqrt"
                elif values["maxFeatures"] == "log2":
                    utils.adaboostParam["maxFeatures"] = "log2"
                elif values["maxFeatures"].isdigit():
                    utils.adaboostParam["maxFeatures"] = int(
                        values["maxFeatures"])
                else:
                    utils.adaboostParam["maxFeatures"] = float(
                        values["maxFeatures"])
                if values["dtRandomState"] == "":
                    utils.adaboostParam["dtRandomState"] = None
                else:
                    utils.adaboostParam["dtRandomState"] = int(
                        values["dtRandomState"])
                if values["maxLeafNodes"] == "":
                    utils.adaboostParam["maxLeafNodes"] = None
                else:
                    utils.adaboostParam["maxLeafNodes"] = int(
                        values["maxLeafNodes"])
                utils.adaboostParam["minImpurityDecrease"] = float(
                    values["minImpurityDecrease"])
                if values["classWeight"] == "":
                    utils.adaboostParam["classWeight"] = None
                else:
                    utils.adaboostParam["classWeight"] = values["classWeight"]
                utils.adaboostParam["ccpAlpha"] = float(values["ccpAlpha"])
                utils.adaboostParam["nEstimators"] = int(values["nEstimators"])
                utils.adaboostParam["learningRate"] = float(
                    values["learningRate"])
                utils.adaboostParam["algorithm"] = values["algorithm"]
                if values["abRandomState"] == "":
                    utils.adaboostParam["abRandomState"] = None
                else:
                    utils.adaboostParam["abRandomState"] = int(
                        values["abRandomState"])
                break
        if event == "default":
            window["criterion"].update("gini")
            window["splitter"].update("best")
            window["maxDepth"].update("")
            window["minSamplesSplit"].update(2)
            window["minSamplesLeaf"].update(1)
            window["minWeightFractionLeaf"].update(0)
            window["maxFeatures"].update("sqrt")
            window["dtRandomState"].update("")
            window["maxLeafNodes"].update("")
            window["minImpurityDecrease"].update(0)
            window["classWeight"].update("")
            window["ccpAlpha"].update(0)
            window["nEstimators"].update(50)
            window["learningRate"].update(1)
            window["algorithm"].update("SAMME.R")
            window["abRandomState"].update("")
    window.Close()


def evaluateModel(model, xTrue, yTrue):
    yPred = model.predict(xTrue)
    tn, fp, fn, tp = confusion_matrix(yTrue, yPred).ravel()
    acc = (tp + tn) / (tp + fp + fn + tn)  # 准确率
    p = tp / (tp + fp)  # 精确率
    sen = tp / (tp + fn)  # 召回率（敏感性）
    spe = tn / (tn + fp)  # 特异性
    f1 = f1_score(yTrue, yPred)  # F1值
    youdenIndex = sen + spe - 1  # 约登指数
    ppv = tp / (tp + fp)  # 阳性预测值
    npv = tn / (tn + fn)  # 阴性预测值
    mcc = matthews_corrcoef(yTrue,
                            yPred)  # 马修斯相关系数（+1代表理想预测，0代表平均随即预测，-1代表反向预测）

    return acc, p, sen, spe, f1, youdenIndex, ppv, npv, mcc


def runRandomForest(x, y, testSize, msgCallBack):
    xTrainRF, xTestRF, yTrainRF, yTestRF = train_test_split(x,
                                                            y,
                                                            test_size=testSize)

    try:
        modelRF = RandomForestClassifier(
            n_estimators=utils.randomForestParam["nEstimators"],
            criterion=utils.randomForestParam["criterion"],
            max_depth=utils.randomForestParam["maxDepth"],
            min_samples_split=utils.randomForestParam["minSamplesSplit"],
            min_samples_leaf=utils.randomForestParam["minSamplesLeaf"],
            min_weight_fraction_leaf=utils.
            randomForestParam["minWeightFractionLeaf"],
            max_features=utils.randomForestParam["maxFeatures"],
            max_leaf_nodes=utils.randomForestParam["maxLeafNodes"],
            min_impurity_decrease=utils.
            randomForestParam["minImpurityDecrease"],
            bootstrap=utils.randomForestParam["bootstrap"],
            oob_score=utils.randomForestParam["oobScore"],
            random_state=utils.randomForestParam["randomState"],
            warm_start=utils.randomForestParam["warmStart"],
            class_weight=utils.randomForestParam["classWeight"],
            ccp_alpha=utils.randomForestParam["ccpAlpha"],
            max_samples=utils.randomForestParam["maxSamples"],
        ).fit(xTrainRF, yTrainRF.values.ravel())
    except Exception as e:
        logger.exception("{}".format(e))
        ui.popupError("{}".format(e))
        return False, None

    accRF, pRF, senRF, speRF, f1RF, youdenIndexRF, ppvRF, npvRF, mccRF = evaluateModel(
        modelRF, xTestRF, yTestRF)

    msgCallBack("*** RF模型的准确率（acc）为: {} ***".format(accRF))
    msgCallBack("*** RF模型的精确率（p）为: {} ***".format(pRF))
    msgCallBack("*** RF模型的召回率/敏感性（sen）为: {} ***".format(senRF))
    msgCallBack("*** RF模型的特异性（spe）为: {} ***".format(speRF))
    msgCallBack("*** RF模型的F1值（f1）为: {} ***".format(f1RF))
    msgCallBack("*** RF模型的约登指数（youdenIndex）为: {} ***".format(youdenIndexRF))
    msgCallBack("*** RF模型的阳性预测值（ppv）为: {} ***".format(ppvRF))
    msgCallBack("*** RF模型的阴性预测值（npv）为: {} ***".format(npvRF))
    msgCallBack("*** RF模型的马修斯相关系数（mcc）为: {} ***".format(mccRF))
    return True, modelRF, xTestRF, yTestRF


def runSVM(x, y, testSize, msgCallBack):
    xTrainSVM, xTestSVM, yTrainSVM, yTestSVM = train_test_split(
        x, y, test_size=testSize)

    try:
        modelSVM = svm.SVC(
            C=utils.svmParam["C"],
            kernel=utils.svmParam["kernel"],
            degree=utils.svmParam["degree"],
            gamma=utils.svmParam["gamma"],
            coef0=utils.svmParam["coef0"],
            shrinking=utils.svmParam["shrinking"],
            probability=utils.svmParam["probability"],
            class_weight=utils.svmParam["classWeight"],
            max_iter=utils.svmParam["maxIter"],
            decision_function_shape=utils.svmParam["decisionFunctionShape"],
            break_ties=utils.svmParam["breakTies"],
            random_state=utils.svmParam["randomState"],
        ).fit(xTrainSVM, yTrainSVM.values.ravel())
    except Exception as e:
        logger.exception("{}".format(e))
        ui.popupError("{}".format(e))
        return False, None

    accSVM, pSVM, senSVM, speSVM, f1SVM, youdenIndexSVM, ppvSVM, npvSVM, mccSVM = evaluateModel(
        modelSVM, xTestSVM, yTestSVM)

    msgCallBack("*** SVM模型的准确率（acc）为: {} ***".format(accSVM))
    msgCallBack("*** SVM模型的精确率（p）为: {} ***".format(pSVM))
    msgCallBack("*** SVM模型的召回率/敏感性（sen）为: {} ***".format(senSVM))
    msgCallBack("*** SVM模型的特异性（spe）为: {} ***".format(speSVM))
    msgCallBack("*** SVM模型的F1值（f1）为: {} ***".format(f1SVM))
    msgCallBack("*** SVM模型的约登指数（youdenIndex）为: {} ***".format(youdenIndexSVM))
    msgCallBack("*** SVM模型的阳性预测值（ppv）为: {} ***".format(ppvSVM))
    msgCallBack("*** SVM模型的阴性预测值（npv）为: {} ***".format(npvSVM))
    msgCallBack("*** SVM模型的马修斯相关系数（mcc）为: {} ***".format(mccSVM))
    return True, modelSVM, xTestSVM, yTestSVM


def runMLP(x, y, testSize, msgCallBack):
    xTrainMLP, xTestMLP, yTrainMLP, yTestMLP = train_test_split(
        x, y, test_size=testSize)

    try:
        modelMLP = MLPClassifier(
            hidden_layer_sizes=utils.mlpParam["hiddenLayerSizes"],
            activation=utils.mlpParam["activation"],
            solver=utils.mlpParam["solver"],
            alpha=utils.mlpParam["alpha"],
            batch_size=utils.mlpParam["batchSize"],
            learning_rate=utils.mlpParam["learningRate"],
            learning_rate_init=utils.mlpParam["learningRateInit"],
            power_t=utils.mlpParam["powerT"],
            max_iter=utils.mlpParam["maxIter"],
            shuffle=utils.mlpParam["shuffle"],
            random_state=utils.mlpParam["randomState"],
            warm_start=utils.mlpParam["warmStart"],
            momentum=utils.mlpParam["momentum"],
            nesterovs_momentum=utils.mlpParam["nesterovsMomentum"],
            early_stopping=utils.mlpParam["earlyStopping"],
            validation_fraction=utils.mlpParam["validationFraction"],
            beta_1=utils.mlpParam["beta1"],
            beta_2=utils.mlpParam["beta2"],
            n_iter_no_change=utils.mlpParam["nIterNoChange"],
            max_fun=utils.mlpParam["maxFun"],
        ).fit(xTrainMLP, yTrainMLP.values.ravel())
    except Exception as e:
        logger.exception("{}".format(e))
        ui.popupError("{}".format(e))
        return False, None

    accMLP, pMLP, senMLP, speMLP, f1MLP, youdenIndexMLP, ppvMLP, npvMLP, mccMLP = evaluateModel(
        modelMLP, xTestMLP, yTestMLP)

    msgCallBack("*** MLP模型的准确率（acc）为: {} ***".format(accMLP))
    msgCallBack("*** MLP模型的精确率（p）为: {} ***".format(pMLP))
    msgCallBack("*** MLP模型的召回率/敏感性（sen）为: {} ***".format(senMLP))
    msgCallBack("*** MLP模型的特异性（spe）为: {} ***".format(speMLP))
    msgCallBack("*** MLP模型的F1值（f1）为: {} ***".format(f1MLP))
    msgCallBack("*** MLP模型的约登指数（youdenIndex）为: {} ***".format(youdenIndexMLP))
    msgCallBack("*** MLP模型的阳性预测值（ppv）为: {} ***".format(ppvMLP))
    msgCallBack("*** MLP模型的阴性预测值（npv）为: {} ***".format(npvMLP))
    msgCallBack("*** MLP模型的马修斯相关系数（mcc）为: {} ***".format(mccMLP))
    return True, modelMLP, xTestMLP, yTestMLP


def runXGBoost(x, y, testSize, msgCallBack):
    # 注意，对于二分类问题，即objective = "binary:logistic"，则参数num_class不用设置，xgboost内部
    # 会调用np.unique(np.asarray(y))判断输入的y的取值类型数量，然后对num_class设置相应的值

    xTrainXGB, xTestXGB, yTrainXGB, yTestXGB = train_test_split(
        x, y, test_size=testSize)

    try:
        modelXGB = XGBClassifier(
            max_depth=utils.xgboostParam["maxDepth"],
            max_leaves=utils.xgboostParam["maxLeaves"],
            max_bin=utils.xgboostParam["maxBin"],
            grow_policy=utils.xgboostParam["growPolicy"],
            learning_rate=utils.xgboostParam["learningRate"],
            n_estimators=utils.xgboostParam["nEstimators"],
            objective=utils.xgboostParam["objective"],
            booster=utils.xgboostParam["booster"],
            tree_method=utils.xgboostParam["treeMethod"],
            gamma=utils.xgboostParam["gamma"],
            min_child_weight=utils.xgboostParam["minChildWeight"],
            max_delta_step=utils.xgboostParam["maxDeltaStep"],
            subsample=utils.xgboostParam["subsample"],
            sampling_method=utils.xgboostParam["samplingMethod"],
            colsample_bytree=utils.xgboostParam["colsampleBytree"],
            colsample_bylevel=utils.xgboostParam["colsampleBylevel"],
            colsample_bynode=utils.xgboostParam["colsampleBynode"],
            scale_pos_weight=utils.xgboostParam["scalePosWeight"],
            base_score=utils.xgboostParam["baseScore"],
            random_state=utils.xgboostParam["randomState"],
        )
    except Exception as e:
        logger.exception("{}".format(e))
        ui.popupError("{}".format(e))
        return False, None

    modelXGB.fit(xTrainXGB, yTrainXGB.values.ravel())

    accXGB, pXGB, senXGB, speXGB, f1XGB, youdenIndexXGB, ppvXGB, npvXGB, mccXGB = evaluateModel(
        modelXGB, xTestXGB, yTestXGB)

    msgCallBack("*** XGB模型的准确率（acc）为: {} ***".format(accXGB))
    msgCallBack("*** XGB模型的精确率（p）为: {} ***".format(pXGB))
    msgCallBack("*** XGB模型的召回率/敏感性（sen）为: {} ***".format(senXGB))
    msgCallBack("*** XGB模型的特异性（spe）为: {} ***".format(speXGB))
    msgCallBack("*** XGB模型的F1值（f1）为: {} ***".format(f1XGB))
    msgCallBack("*** XGB模型的约登指数（youdenIndex）为: {} ***".format(youdenIndexXGB))
    msgCallBack("*** XGB模型的阳性预测值（ppv）为: {} ***".format(ppvXGB))
    msgCallBack("*** XGB模型的阴性预测值（npv）为: {} ***".format(npvXGB))
    msgCallBack("*** XGB模型的马修斯相关系数（mcc）为: {} ***".format(mccXGB))
    return True, modelXGB, xTestXGB, yTestXGB


def runAdaBoost(x, y, testSize, msgCallBack):
    xTrainADA, xTestADA, yTrainADA, yTestADA = train_test_split(
        x, y, test_size=testSize)

    try:
        modelADA = AdaBoostClassifier(
            DecisionTreeClassifier(
                criterion=utils.adaboostParam["criterion"],
                splitter=utils.adaboostParam["splitter"],
                max_depth=utils.adaboostParam["maxDepth"],
                min_samples_split=utils.adaboostParam["minSamplesSplit"],
                min_samples_leaf=utils.adaboostParam["minSamplesLeaf"],
                min_weight_fraction_leaf=utils.
                adaboostParam["minWeightFractionLeaf"],
                max_features=utils.adaboostParam["maxFeatures"],
                random_state=utils.adaboostParam["dtRandomState"],
                max_leaf_nodes=utils.adaboostParam["maxLeafNodes"],
                min_impurity_decrease=utils.
                adaboostParam["minImpurityDecrease"],
                class_weight=utils.adaboostParam["classWeight"],
                ccp_alpha=utils.adaboostParam["ccpAlpha"],
            ),
            n_estimators=utils.adaboostParam["nEstimators"],
            learning_rate=utils.adaboostParam["learningRate"],
            algorithm=utils.adaboostParam["algorithm"],
            random_state=utils.adaboostParam["abRandomState"],
        )
        modelADA.fit(xTrainADA, yTrainADA.values.ravel())
    except Exception as e:
        logger.exception("{}".format(e))
        ui.popupError("{}".format(e))
        return False, None

    accADA, pADA, senADA, speADA, f1ADA, youdenIndexADA, ppvADA, npvADA, mccADA = evaluateModel(
        modelADA, xTestADA, yTestADA)

    msgCallBack("*** ADA模型的准确率（acc）为: {} ***".format(accADA))
    msgCallBack("*** ADA模型的精确率（p）为: {} ***".format(pADA))
    msgCallBack("*** ADA模型的召回率/敏感性（sen）为: {} ***".format(senADA))
    msgCallBack("*** ADA模型的特异性（spe）为: {} ***".format(speADA))
    msgCallBack("*** ADA模型的F1值（f1）为: {} ***".format(f1ADA))
    msgCallBack("*** ADA模型的约登指数（youdenIndex）为: {} ***".format(youdenIndexADA))
    msgCallBack("*** ADA模型的阳性预测值（ppv）为: {} ***".format(ppvADA))
    msgCallBack("*** ADA模型的阴性预测值（npv）为: {} ***".format(npvADA))
    msgCallBack("*** ADA模型的马修斯相关系数（mcc）为: {} ***".format(mccADA))
    return True, modelADA, xTestADA, yTestADA


showClassifierParamWindow = {
    "randomForestParam": showRandomForestParamWindow,
    "svmParam": showSVMParamWindow,
    "mlpParam": showMLPParamWindow,
    "xgboostParam": showXGBoostParamWindow,
    "adaboostParam": showAdaBoostParamWindow,
}

classifier = {
    "randomForest": runRandomForest,
    "svm": runSVM,
    "mlp": runMLP,
    "xgboost": runXGBoost,
    "adaboost": runAdaBoost,
}
