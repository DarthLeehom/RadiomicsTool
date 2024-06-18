import featureconfigwindow as fcw
import utils
import ui
import re
import preprocessing as pp
import featureselection as fs
import classify
from mainprocedure import FeatureExtractor, FeatureSelectorAndClassifier

mainWindow = ui.createMainWindow("影像组学特征分析建模软件", icon="./radiomicstool.ico")
enabledFeatures = utils.getAllFeatures()
fe = FeatureExtractor()
fsc = FeatureSelectorAndClassifier()


def setUIEnabled(enabled: bool):
    mainWindow["extract"].update(disabled=~enabled)
    mainWindow["runFeaturesSelectAndClassify"].update(disabled=~enabled)
    mainWindow["showFeatures"].update(disabled=~enabled)
    mainWindow["saveFeatures"].update(disabled=~enabled)
    mainWindow["saveModel"].update(disabled=~enabled)
    mainWindow["lassoFeatureWeight"].update(disabled=~enabled)
    mainWindow["lassoMSE"].update(disabled=~enabled)
    mainWindow["lassoFeatureTrendCurve"].update(disabled=~enabled)
    mainWindow["biplot"].update(disabled=~enabled)
    mainWindow["roc"].update(disabled=~enabled)
    mainWindow["heatmap"].update(disabled=~enabled)


while True:
    event, vals = mainWindow.read(timeout=500)
    if event in (None, "Exit"):
        break
    if event == "allPreprocessType":
        state = vals["allPreprocessType"]
        mainWindow["square"].update(state)
        mainWindow["squareRoot"].update(state)
        mainWindow["logarithm"].update(state)
        mainWindow["exponential"].update(state)
        mainWindow["LoG"].update(state)
        mainWindow["wavelet"].update(state)
    if event == "allFeatureType":
        state = vals["allFeatureType"]
        mainWindow["shape"].update(state)
        mainWindow["firstorder"].update(state)
        mainWindow["glcm"].update(state)
        mainWindow["glrlm"].update(state)
        mainWindow["glszm"].update(state)
        mainWindow["gldm"].update(state)
        mainWindow["ngtdm"].update(state)
    if event == "allFeatureSelection":
        state = vals["allFeatureSelection"]
        mainWindow["TTest"].update(state)
        mainWindow["Lasso"].update(state)
        mainWindow["PCA"].update(state)
        mainWindow["MUIF"].update(state)
        mainWindow["chi2"].update(state)
        mainWindow["mRMR"].update(state)
    if (event == "shapeCfg" or event == "firstorderCfg" or event == "glcmCfg"
            or event == "glrlmCfg" or event == "glszmCfg" or event == "gldmCfg"
            or event == "ngtdmCfg"):
        enabledFeatures[event[:-3]] = fcw.showFeatureConfigWindow(
            event[:-3], enabledFeatures[event[:-3]])
    if (event == "ttestParam" or event == "lassoParam" or event == "pcaParam"
            or event == "muifParam" or event == "chi2Param"
            or event == "mRMRParam"):
        fs.showFeatureSelectionParamWindow[event]()
    if (event == "randomForestParam" or event == "svmParam"
            or event == "mlpParam" or event == "xgboostParam"
            or event == "adaboostParam"):
        classify.showClassifierParamWindow[event]()
    if event == "updateExtractProgressbar":
        progress = vals[event]["progress"]
        allProgressNum = vals[event]["allProgressNum"]
        mainWindow["progressbar"].update(progress, allProgressNum)
    if event == "updateState":
        utils.setState(vals[event]["isBusy"])
        if utils.isBusy():
            setUIEnabled(False)
        else:
            setUIEnabled(True)
    if event == "showMsg":
        msg = vals[event]["msg"]
        ui.popupMsg(msg)
    if event == "extract":
        preprocess = pp.checkPreprocessingType(vals)
        if not preprocess[0]:
            continue

        for featureType in list(enabledFeatures.keys()):
            if not vals[featureType]:
                del enabledFeatures[featureType]
        if not enabledFeatures:
            ui.popupError("Please select a feature!")
            continue

        mainWindow["extractMsg"].update("")

        dataPath = vals["dataFolder"].strip()

        print("dataPath: ", dataPath)

        if dataPath == "":
            ui.popupError("Data path cannot be empty!")
            continue

        fe.runExtract(
            mainWindow=mainWindow,
            dataPath=dataPath,
            preprocess=preprocess[1],
            enabledFeatures=enabledFeatures,
        )
    if event == "showFeatures":
        fe.showFeatures()
    if event == "saveFeatures":
        fe.saveFeatures(mainWindow=mainWindow)
    if event == "saveModel":
        fsc.saveModel()
    if event == "lassoFeatureWeight" and vals["Lasso"]:
        fsc.showLassoFeatureWeight()
    if event == "lassoMSE" and vals["Lasso"]:
        fsc.showLassoMSE()
    if event == "lassoFeatureTrendCurve" and vals["Lasso"]:
        fsc.showLassoFeatureTrendCurve()
    if event == "biplot" and vals["PCA"]:
        fsc.showPCABiplot()
    if event == "heatmap":
        fsc.showHeatmap()
    if event == "roc":
        fsc.showROC()
    if event == "showExtractMsg":
        txt = vals["extractMsg"] + "\n" + vals[event]["msg"]
        mainWindow["extractMsg"].update(txt)
    if event == "showFeautresSelectAndClassifyMsg":
        txt = vals["featuresSelectAndClassifyMsg"] + "\n" + vals[event]["msg"]
        mainWindow["featuresSelectAndClassifyMsg"].update(txt)
    if event == "runFeaturesSelectAndClassify":
        if re.match("^(0\.\d+)$", vals["testSize"]) is None:
            ui.popupError("testSize Error, please check your input!")
            continue

        mainWindow["featuresSelectAndClassifyMsg"].update("")

        testSize = float(vals["testSize"])

        featurePath = vals["featureFolder"].strip()

        featureSelection = {}
        featureSelection["useTTest"] = bool(vals["TTest"])
        featureSelection["useLasso"] = bool(vals["Lasso"])
        featureSelection["usePCA"] = bool(vals["PCA"])
        featureSelection["useMUIF"] = bool(vals["MUIF"])
        featureSelection["useChi2"] = bool(vals["chi2"])
        featureSelection["usemRMR"] = bool(vals["mRMR"])

        if vals["randomForest"]:
            classifierName = "randomForest"
        elif vals["svm"]:
            classifierName = "svm"
        elif vals["mlp"]:
            classifierName = "mlp"
        elif vals["xgboost"]:
            classifierName = "xgboost"
        elif vals["adaboost"]:
            classifierName = "adaboost"

        fsc.runFeaturesSelectAndClassify(
            mainWindow=mainWindow,
            featurePath=featurePath,
            featureSelection=featureSelection,
            testSize=testSize,
            classifierName=classifierName,
        )

mainWindow.Close()
