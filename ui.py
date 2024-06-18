import PySimpleGUI as sg
import utils


def popupMsg(msg: str):
    sg.popup(msg)


def popupError(msg: str):
    sg.popup_error(msg)


def __createBottomLayout():
    return [
        [sg.Text("_" * 80)],
        [
            sg.Button("确定",
                      key="ok",
                      enable_events=True,
                      disabled=utils.isBusy()),
            sg.Button("取消", key="cancel", enable_events=True),
        ],
    ]


def __createMainWindowLayout():
    freatureExtractTab = [
        [
            sg.Text(
                "选择样本数据文件夹",
                size=(35, 1),
                font=("Helvetica", 12),
                text_color="yellow",
            )
        ],
        [
            sg.Text(
                "数据文件夹",
                size=(12, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                r"G:/NFYYProjects/radiomics/radiomics/data/BraTS19/",
                key="dataFolder",
                size=(50, 1),
            ),
            sg.FolderBrowse(button_text="浏览", key="dataFolderBrowse"),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Text(
                "预处理类型",
                size=(35, 1),
                font=("Helvetica", 12),
                text_color="yellow",
            ),
            sg.Checkbox("全部", key="allPreprocessType", enable_events=True),
        ],
        [
            sg.Checkbox("平方", key="square", enable_events=True, default=False),
            sg.Checkbox("平方根",
                        key="squareRoot",
                        enable_events=True,
                        default=False),
            sg.Checkbox("对数",
                        key="logarithm",
                        enable_events=True,
                        default=False),
            sg.Checkbox("指数",
                        key="exponential",
                        enable_events=True,
                        default=False),
        ],
        [
            sg.Checkbox("高斯拉普拉斯", key="LoG", enable_events=True,
                        default=False),
            sg.Text("sigma",
                    size=(5, 1),
                    auto_size_text=False,
                    justification="left"),
            sg.InputText(
                r"2,3,4,5",
                key="sigma",
                tooltip="Enter positive integers separated by commas",
            ),  # sigma: List of floats or integers, must be greater than 0. Filter width (mm) to use for the Gaussian kernel
        ],
        [
            sg.Checkbox("小波", key="wavelet", enable_events=True,
                        default=False),
            sg.Text("type",
                    size=(3, 1),
                    auto_size_text=False,
                    justification="left"),
            sg.InputCombo(
                utils.getWaveletTypeList(),
                default_value="coif1",
                size=(8, 10),
                readonly=True,
                key="waveletType",
            ),
            sg.Text("startLevel",
                    size=(7, 1),
                    auto_size_text=False,
                    justification="left"),
            sg.InputText(
                r"0",
                key="startLevel",
                size=(8, 1),
                tooltip="An integer greater than or equal to 0",
            ),
            sg.Text("level",
                    size=(3, 1),
                    auto_size_text=False,
                    justification="left"),
            sg.InputText(
                r"1",
                key="level",
                size=(8, 1),
                tooltip="An integer greater than 0",
            ),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Text(
                "特征类型",
                size=(35, 1),
                font=("Helvetica", 12),
                text_color="yellow",
            ),
            sg.Checkbox("全部", key="allFeatureType", enable_events=True),
        ],
        [
            sg.Checkbox("形态", key="shape", enable_events=True, default=True),
            sg.Button("配置", key="shapeCfg", enable_events=True),
            sg.Checkbox("一阶统计",
                        key="firstorder",
                        enable_events=True,
                        default=True),
            sg.Button("配置", key="firstorderCfg", enable_events=True),
            sg.Checkbox("灰度共生矩阵", key="glcm", enable_events=True,
                        default=True),
            sg.Button("配置", key="glcmCfg", enable_events=True),
            sg.Checkbox("灰度行程（游程）矩阵",
                        key="glrlm",
                        enable_events=True,
                        default=True),
            sg.Button("配置", key="glrlmCfg", enable_events=True),
        ],
        [
            sg.Checkbox("灰度区域大小矩阵",
                        key="glszm",
                        enable_events=True,
                        default=True),
            sg.Button("配置", key="glszmCfg", enable_events=True),
            sg.Checkbox("灰度依赖矩阵", key="gldm", enable_events=True,
                        default=True),
            sg.Button("配置", key="gldmCfg", enable_events=True),
            sg.Checkbox("邻域灰度差矩阵",
                        key="ngtdm",
                        enable_events=True,
                        default=True),
            sg.Button("配置", key="ngtdmCfg", enable_events=True),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Button("显示特征",
                      key="showFeatures",
                      enable_events=True,
                      size=(13, 1)),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Text("提取进度",
                    key="progress",
                    font=("Helvetica", 12),
                    text_color="yellow")
        ],
        [
            sg.ProgressBar(100,
                           orientation="h",
                           size=(50, 20),
                           key="progressbar")
        ],
        [
            sg.Multiline("",
                         key="extractMsg",
                         size=(78, 6),
                         autoscroll=True,
                         disabled=True)
        ],
        [
            sg.Button("开始提取", key="extract", enable_events=True, size=(9, 1)),
            sg.Button("保存特征", key="saveFeatures", enable_events=True),
        ],
    ]
    featuresSelectAndClassifyTab = [
        [
            sg.Text(
                "选择特征文件夹",
                size=(35, 1),
                font=("Helvetica", 12),
                text_color="yellow",
            )
        ],
        [
            sg.Text(
                "特征文件夹",
                size=(13, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                r"G:/NFYYProjects/radiomics/radiomics/features",
                key="featureFolder",
                size=(50, 1),
            ),
            sg.FolderBrowse(button_text="浏览", key="featureFolderBrowse"),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Text(
                "特征筛选",
                size=(35, 1),
                font=("Helvetica", 12),
                text_color="yellow",
            ),
            sg.Checkbox("全部", key="allFeatureSelection", enable_events=True),
        ],
        [
            sg.Checkbox("T测试", key="TTest", enable_events=True, default=False),
            sg.Button("参数设置", key="ttestParam", enable_events=True),
            sg.Checkbox("卡方检测", key="chi2", enable_events=True, default=False),
            sg.Button("参数设置", key="chi2Param", enable_events=True),
            sg.Checkbox("互信息（MUIF）",
                        key="MUIF",
                        enable_events=True,
                        default=False),
            sg.Button("参数设置", key="muifParam", enable_events=True),
        ],
        [
            sg.Checkbox("主成分分析（PCA）",
                        key="PCA",
                        enable_events=True,
                        default=False),
            sg.Button("参数设置", key="pcaParam", enable_events=True),
            sg.Checkbox("最大相关最小冗余（mRMR）",
                        key="mRMR",
                        enable_events=True,
                        default=False),
            sg.Button("参数设置", key="mRMRParam", enable_events=True),
            sg.Checkbox("Lasso",
                        key="Lasso",
                        enable_events=True,
                        default=False),
            sg.Button("参数设置", key="lassoParam", enable_events=True),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Text(
                "分类器",
                size=(25, 1),
                font=("Helvetica", 12),
                text_color="yellow",
            )
        ],
        [
            sg.Text("测试集大小",
                    size=(8, 1),
                    auto_size_text=False,
                    justification="left"),
            sg.InputText(
                "0.3",
                key="testSize",
                tooltip="0 < float < 1",
                size=(5, 1),
            ),
        ],
        [
            sg.Radio(
                "随机森林",
                "classifier",
                key="randomForest",
                enable_events=True,
                default=True,
            ),
            sg.Button("参数设置", key="randomForestParam", enable_events=True),
            sg.Radio("支持向量机（SVM）",
                     "classifier",
                     key="svm",
                     enable_events=True,
                     default=False),
            sg.Button("参数设置", key="svmParam", enable_events=True),
            sg.Radio("多层感知器（MLP）",
                     "classifier",
                     key="mlp",
                     enable_events=True,
                     default=False),
            sg.Button("参数设置", key="mlpParam", enable_events=True),
        ],
        [
            sg.Radio(
                "XGBoost",
                "classifier",
                key="xgboost",
                enable_events=True,
                default=False,
            ),
            sg.Button("参数设置", key="xgboostParam", enable_events=True),
            sg.Radio(
                "AdaBoost",
                "classifier",
                key="adaboost",
                enable_events=True,
                default=False,
            ),
            sg.Button("参数设置", key="adaboostParam", enable_events=True),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Text(
                "画图",
                size=(25, 1),
                font=("Helvetica", 12),
                text_color="yellow",
            )
        ],
        [
            sg.Button(
                "Lasso feature weight",
                key="lassoFeatureWeight",
                size=(16, 1),
                enable_events=True,
            ),
            sg.Button(
                "Lasso mse",
                key="lassoMSE",
                size=(12, 1),
                enable_events=True,
                tooltip="Lasso mean square error",
            ),
            sg.Button(
                "Lasso feature trend curve",
                key="lassoFeatureTrendCurve",
                size=(19, 1),
                enable_events=True,
            ),
        ],
        [
            sg.Button(
                "PCA biplot",
                key="biplot",
                size=(12, 1),
                enable_events=True,
            ),
            sg.Button(
                "ROC",
                key="roc",
                size=(5, 1),
                enable_events=True,
            ),
            sg.Button(
                "Feature heatmap",
                key="heatmap",
                size=(15, 1),
                enable_events=True,
            ),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Multiline(
                "",
                key="featuresSelectAndClassifyMsg",
                size=(78, 6),
                autoscroll=True,
                disabled=True,
            )
        ],
        [
            sg.Button(
                "运行",
                key="runFeaturesSelectAndClassify",
                size=(8, 1),
                enable_events=True,
            ),
            sg.Button(
                "保存模型",
                key="saveModel",
                size=(8, 1),
                enable_events=True,
            ),
        ],
    ]
    return [
        [
            sg.TabGroup([[
                sg.Tab(
                    title="特征提取",
                    layout=freatureExtractTab,
                ),
                sg.Tab(
                    title="特征筛选与建模",
                    layout=featuresSelectAndClassifyTab,
                ),
            ]])
        ],
        [sg.Text("_" * 80)],
        [
            sg.Exit(button_text="退出", key="exit"),
        ],
    ]


def __createFeatureLayout(featureType: str, enabledFeatures: list):
    featureNames = utils.getAllFeatures()[featureType]
    if not featureNames:
        sg.popup_error("Undefined feature type!")
        return []

    featureCount = len(featureNames)
    columnCount = 5
    rowCount = featureCount // columnCount
    lastRowCount = featureCount % columnCount
    featureLayout = []

    for i in range(rowCount):
        checkboxList = []
        for name in featureNames[(i * columnCount):((i + 1) * columnCount)]:
            checkboxList.append(
                sg.Checkbox(
                    name,
                    key=name,
                    enable_events=True,
                    default=True if name in enabledFeatures else False,
                ))
        featureLayout.append(checkboxList)

    if lastRowCount > 0:
        checkboxList = []
        for name in featureNames[(rowCount * columnCount):]:
            checkboxList.append(
                sg.Checkbox(
                    name,
                    key=name,
                    enable_events=True,
                    default=True if name in enabledFeatures else False,
                ))
        featureLayout.append(checkboxList)

    headLayout = [
        [
            sg.Checkbox("全部", key="all", enable_events=True, default=False),
        ],
        [sg.Text("_" * 80)],
    ]
    return headLayout + featureLayout


def __createTTestParamLayout():
    return [
        [
            sg.Text(
                "Levene",
                size=(35, 1),
                font=("Helvetica", 12),
                text_color="yellow",
            )
        ],
        [
            sg.Text(
                "center",
                size=(12, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["mean", "median", "trimmed"],
                default_value=utils.ttestParam["leveneCenter"],
                key="center",
                readonly=True,
                size=(8, 6),
            ),
            sg.Text(
                "proportiontocut",
                size=(12, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.ttestParam["leveneProportiontocut"],
                key="proportiontocut",
                tooltip="Float, valid range is [0, 1]",
                size=(10, 1),
            ),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Text(
                "TTest",
                size=(35, 1),
                font=("Helvetica", 12),
                text_color="yellow",
            )
        ],
        [
            sg.Text(
                "axis",
                size=(6, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(utils.ttestParam["ttestAxis"],
                         key="axis",
                         tooltip="Int",
                         size=(10, 1)),
            sg.Text(
                "nan_policy",
                size=(8, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["propagate", "raise", "omit"],
                default_value=utils.ttestParam["ttestNanPolicy"],
                key="nanPolicy",
                size=(10, 6),
                readonly=True,
            ),
            sg.Text(
                "permutations",
                size=(10, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.ttestParam["ttestPermutations"],
                key="permutations",
                tooltip="Empty or non-negative int",
                size=(10, 1),
            ),
            sg.Text(
                "randomState",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.ttestParam["ttestRandomState"],
                key="randomState",
                tooltip="Empty or int",
                size=(10, 1),
            ),
        ],
        [
            sg.Text(
                "alternative",
                size=(8, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["two-sided", "less", "greater"],
                default_value=utils.ttestParam["ttestAlternative"],
                key="alternative",
                size=(10, 6),
                readonly=True,
            ),
            sg.Text(
                "trim",
                size=(5, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.ttestParam["ttestTrim"],
                key="trim",
                tooltip="Float, valid range is [0, 0.5)",
                size=(10, 1),
            ),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Button("默认", key="default", enable_events=True),
        ],
    ]


def __createLassoParamLayout():
    return [
        [
            sg.Text(
                "Alphas",
                size=(35, 1),
                font=("Helvetica", 12),
                text_color="yellow",
            )
        ],
        [
            sg.Checkbox(
                "custom alphas",
                key="customAlphas",
                enable_events=True,
                default=utils.lassoParam["customAlphas"],
            ),
        ],
        [
            sg.Text(
                "start",
                size=(3, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.lassoParam["start"],
                key="start",
                tooltip="Int, stop > start",
                size=(10, 1),
            ),
            sg.Text(
                "stop",
                size=(3, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.lassoParam["stop"],
                key="stop",
                tooltip="Int, stop > start",
                size=(10, 1),
            ),
            sg.Text(
                "num",
                size=(3, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.lassoParam["num"],
                key="num",
                tooltip="int > 0",
                size=(10, 1),
            ),
            sg.Text(
                "base",
                size=(4, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.lassoParam["base"],
                key="base",
                tooltip="int > 0",
                size=(10, 1),
            ),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Text(
                "Lasso Param",
                size=(35, 1),
                font=("Helvetica", 12),
                text_color="yellow",
            )
        ],
        [
            sg.Text(
                "eps",
                size=(3, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.lassoParam["eps"],
                key="eps",
                tooltip="0 <= float <= 1",
                size=(10, 1),
            ),
            sg.Text(
                "n_alphas",
                size=(7, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.lassoParam["nAlphas"],
                key="nAlphas",
                tooltip="int > 0",
                size=(10, 1),
            ),
            sg.Checkbox(
                "fit_intercept",
                key="fitIntercept",
                enable_events=True,
                default=utils.lassoParam["fitIntercept"],
            ),
            sg.Text(
                "max_iter",
                size=(6, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.lassoParam["maxIter"],
                key="maxIter",
                tooltip="int > 0",
                size=(10, 1),
            ),
        ],
        [
            sg.Text(
                "cv",
                size=(2, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.lassoParam["cv"],
                key="cv",
                tooltip="None or int > 0",
                size=(10, 1),
            ),
            sg.Text(
                "randomState",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.lassoParam["randomState"],
                key="randomState",
                tooltip="None or int >= 0",
                size=(10, 1),
            ),
            sg.Text(
                "selection",
                size=(7, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["cyclic", "random"],
                default_value="cyclic",
                size=(8, 4),
                readonly=True,
                key="selection",
            ),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Button("默认", key="default", enable_events=True),
        ],
    ]


def __createPCAParamLayout():
    return [
        [
            sg.Text(
                "nComponents",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.pcaParam["nComponents"],
                key="nComponents",
                tooltip="None or int > 0 or 0 < float < 1",
                size=(10, 1),
            ),
            sg.Checkbox(
                "whiten",
                key="whiten",
                enable_events=True,
                default=utils.pcaParam["whiten"],
                size=(5, 1),
            ),
            sg.Text(
                "svdSolver",
                size=(7, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["auto", "full", "arpack", "randomized"],
                key="svdSolver",
                default_value=utils.pcaParam["svdSolver"],
                readonly=True,
                size=(8, 6),
            ),
            sg.Text(
                "iteratedPower",
                size=(10, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.pcaParam["iteratedPower"],
                key="iteratedPower",
                tooltip='"auto" or int >= 0',
                size=(10, 1),
            ),
        ],
        [
            sg.Text(
                "nOversamples",
                size=(10, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.pcaParam["nOversamples"],
                key="nOversamples",
                tooltip="int > 0",
                size=(10, 1),
            ),
            sg.Text(
                "powerIterationNormalizer",
                size=(18, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["auto", "QR", "LU", "none"],
                key="powerIterationNormalizer",
                default_value=utils.pcaParam["powerIterationNormalizer"],
                readonly=True,
                size=(8, 6),
            ),
            sg.Text(
                "randomState",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.pcaParam["randomState"],
                key="randomState",
                tooltip="None or int >= 0",
                size=(10, 1),
            ),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Button("默认", key="default", enable_events=True),
        ],
    ]


def __createMUIFParamLayout():
    return [
        [
            sg.Radio(
                "useKBest",
                "muif",
                key="useKBest",
                enable_events=True,
                default=utils.muifParam["useKBest"],
            ),
            sg.Text(
                "k",
                size=(1, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.muifParam["k"],
                key="k",
                tooltip='"all" or int > 0',
                size=(10, 1),
            ),
            sg.Radio(
                "usePercentile",
                "muif",
                key="usePercentile",
                enable_events=True,
                default=utils.muifParam["usePercentile"],
            ),
            sg.Text(
                "percentile",
                size=(7, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.muifParam["percentile"],
                key="percentile",
                tooltip="0 < int <= 100",
                size=(10, 1),
            ),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Button("默认", key="default", enable_events=True),
        ],
    ]


def __createChi2ParamLayout():
    return [
        [
            sg.Radio(
                "useKBest",
                "chi2",
                key="useKBest",
                enable_events=True,
                default=utils.chi2Param["useKBest"],
            ),
            sg.Text(
                "k",
                size=(1, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.chi2Param["k"],
                key="k",
                tooltip='"all" or int > 0',
                size=(10, 1),
            ),
            sg.Radio(
                "usePercentile",
                "chi2",
                key="usePercentile",
                enable_events=True,
                default=utils.chi2Param["usePercentile"],
            ),
            sg.Text(
                "percentile",
                size=(7, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.chi2Param["percentile"],
                key="percentile",
                tooltip="0 < int <= 100",
                size=(10, 1),
            ),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Button("默认", key="default", enable_events=True),
        ],
    ]


def __createmRMRParamLayout():
    return [
        [
            sg.Text(
                "method",
                size=(5, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["MIQ", "MID"],
                key="method",
                default_value=utils.mRMRParam["method"],
                readonly=True,
                size=(8, 6),
            ),
            sg.Text(
                "nfeats",
                size=(4, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.mRMRParam["nfeats"],
                key="nfeats",
                tooltip="int > 0",
                size=(10, 1),
            ),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Button("默认", key="default", enable_events=True),
        ],
    ]


def createMainWindow(title: str, icon: str):
    layout = __createMainWindowLayout()
    return sg.Window(title, layout, icon=icon)


def createFeatureConfigWindow(featureType: str, enabledFeatures: list,
                              title: str):
    layout = __createFeatureLayout(featureType, enabledFeatures)
    if not layout:
        return None

    bottomLayout = __createBottomLayout()
    layout = layout + bottomLayout

    return sg.Window(title, layout, modal=True)


def createTTestParamWindow():
    layout = __createTTestParamLayout() + __createBottomLayout()
    return sg.Window("T测试参数设置", layout, modal=True)


def createLassoParamWindow():
    layout = __createLassoParamLayout() + __createBottomLayout()
    return sg.Window("Lasso参数设置", layout, modal=True)


def createPCAParamWindow():
    layout = __createPCAParamLayout() + __createBottomLayout()
    return sg.Window("PCA参数设置", layout, modal=True)


def createMUIFParamWindow():
    layout = __createMUIFParamLayout() + __createBottomLayout()
    return sg.Window("MUIF参数设置", layout, modal=True)


def createChi2ParamWindow():
    layout = __createChi2ParamLayout() + __createBottomLayout()
    return sg.Window("卡方检测参数设置", layout, modal=True)


def createmRMRParamWindow():
    layout = __createmRMRParamLayout() + __createBottomLayout()
    return sg.Window("mRMR参数设置", layout, modal=True)


def __createRandomForestParamLayout():
    return [
        [
            sg.Text(
                "nEstimators",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.randomForestParam["nEstimators"],
                key="nEstimators",
                tooltip="int > 0",
                size=(10, 1),
            ),
            sg.Text(
                "criterion",
                size=(6, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["gini", "entropy", "log_loss"],
                key="criterion",
                default_value=utils.randomForestParam["criterion"],
                readonly=True,
                size=(8, 6),
            ),
            sg.Text(
                "maxDepth",
                size=(7, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.randomForestParam["maxDepth"],
                key="maxDepth",
                tooltip="None or int > 0",
                size=(10, 1),
            ),
            sg.Text(
                "minSamplesSplit",
                size=(12, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.randomForestParam["minSamplesSplit"],
                key="minSamplesSplit",
                tooltip="int > 0 or 0 < float <= 1",
                size=(10, 1),
            ),
        ],
        [
            sg.Text(
                "minSamplesLeaf",
                size=(12, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.randomForestParam["minSamplesLeaf"],
                key="minSamplesLeaf",
                tooltip="int > 0 or 0 < float <= 1",
                size=(10, 1),
            ),
            sg.Text(
                "minWeightFractionLeaf",
                size=(17, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.randomForestParam["minWeightFractionLeaf"],
                key="minWeightFractionLeaf",
                tooltip="0 <= float <= 1",
                size=(10, 1),
            ),
            sg.Text(
                "maxFeatures",
                size=(10, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.randomForestParam["maxFeatures"],
                key="maxFeatures",
                tooltip='None or "sqrt" or "log2" or int > 0 or 0 < float <= 1',
                size=(10, 1),
            ),
            sg.Text(
                "maxLeafNodes",
                size=(11, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.randomForestParam["maxLeafNodes"],
                key="maxLeafNodes",
                tooltip="None or int > 0",
                size=(10, 1),
            ),
        ],
        [
            sg.Text(
                "minImpurityDecrease",
                size=(15, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.randomForestParam["minImpurityDecrease"],
                key="minImpurityDecrease",
                tooltip="float >= 0",
                size=(10, 1),
            ),
            sg.Checkbox(
                "bootstrap",
                key="bootstrap",
                enable_events=True,
                default=utils.randomForestParam["bootstrap"],
                size=(7, 1),
            ),
            sg.Checkbox(
                "oobScore",
                key="oobScore",
                enable_events=True,
                default=utils.randomForestParam["oobScore"],
                size=(7, 1),
            ),
            sg.Text(
                "randomState",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.randomForestParam["randomState"],
                key="randomState",
                tooltip="None or int >= 0",
                size=(10, 1),
            ),
        ],
        [
            sg.Checkbox(
                "warmStart",
                key="warmStart",
                enable_events=True,
                default=utils.randomForestParam["warmStart"],
                size=(7, 1),
            ),
            sg.Text(
                "classWeight",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["", "balanced", "balanced_subsample"],
                key="classWeight",
                default_value=utils.randomForestParam["classWeight"],
                readonly=True,
                size=(8, 6),
            ),
            sg.Text(
                "ccpAlpha",
                size=(7, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.randomForestParam["ccpAlpha"],
                key="ccpAlpha",
                tooltip="float >= 0",
                size=(10, 1),
            ),
            sg.Text(
                "maxSamples",
                size=(10, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.randomForestParam["maxSamples"],
                key="maxSamples",
                tooltip="None or int > 0 or 0 < float <= 1",
                size=(10, 1),
            ),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Button("默认", key="default", enable_events=True),
        ],
    ]


def __createSVMParamLayout():
    return [
        [
            sg.Text(
                "C",
                size=(1, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.svmParam["C"],
                key="C",
                tooltip="float > 0",
                size=(10, 1),
            ),
            sg.Text(
                "kernel",
                size=(4, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["linear", "poly", "rbf", "sigmoid", "precomputed"],
                key="kernel",
                default_value=utils.svmParam["kernel"],
                readonly=True,
                size=(8, 6),
            ),
            sg.Text(
                "degree",
                size=(5, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.svmParam["degree"],
                key="degree",
                tooltip="int >= 0",
                size=(10, 1),
            ),
            sg.Text(
                "gamma",
                size=(5, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.svmParam["gamma"],
                key="gamma",
                tooltip='"scale" or "auto" or float >= 0',
                size=(10, 1),
            ),
        ],
        [
            sg.Text(
                "coef0",
                size=(4, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.svmParam["coef0"],
                key="coef0",
                tooltip="float >= 0",
                size=(10, 1),
            ),
            sg.Checkbox(
                "shrinking",
                key="shrinking",
                enable_events=True,
                default=utils.svmParam["shrinking"],
                size=(7, 1),
            ),
            sg.Checkbox(
                "probability",
                key="probability",
                enable_events=True,
                default=utils.svmParam["probability"],
                size=(7, 1),
            ),
            sg.Text(
                "classWeight",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["", "balanced"],
                key="classWeight",
                default_value=utils.svmParam["classWeight"],
                readonly=True,
                size=(8, 6),
            ),
        ],
        [
            sg.Text(
                "maxIter",
                size=(6, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.svmParam["maxIter"],
                key="maxIter",
                tooltip="-1 or int > 0",
                size=(10, 1),
            ),
            sg.Text(
                "decisionFunctionShape",
                size=(17, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["ovo", "ovr"],
                key="decisionFunctionShape",
                default_value=utils.svmParam["decisionFunctionShape"],
                readonly=True,
                size=(8, 6),
            ),
            sg.Checkbox(
                "breakTies",
                key="breakTies",
                enable_events=True,
                default=utils.svmParam["breakTies"],
                size=(7, 1),
            ),
            sg.Text(
                "randomState",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.svmParam["randomState"],
                key="randomState",
                tooltip="None or int >= 0",
                size=(10, 1),
            ),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Button("默认", key="default", enable_events=True),
        ],
    ]


def __createMLPParamLayout():
    return [
        [
            sg.Text(
                "hiddenLayerSizes",
                size=(13, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                ",".join(map(str, utils.mlpParam["hiddenLayerSizes"])),
                key="hiddenLayerSizes",
                tooltip="Positive integers separated by commas",
                size=(10, 1),
            ),
            sg.Text(
                "activation",
                size=(7, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["identity", "logistic", "tanh", "relu"],
                key="activation",
                default_value=utils.mlpParam["activation"],
                readonly=True,
                size=(8, 6),
            ),
            sg.Text(
                "solver",
                size=(4, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["lbfgs", "sgd", "adam"],
                key="solver",
                default_value=utils.mlpParam["solver"],
                readonly=True,
                size=(8, 6),
            ),
            sg.Text(
                "alpha",
                size=(4, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.mlpParam["alpha"],
                key="alpha",
                tooltip="float > 0",
                size=(10, 1),
            ),
            sg.Text(
                "batchSize",
                size=(7, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.mlpParam["batchSize"],
                key="batchSize",
                tooltip="auto or int > 0",
                size=(10, 1),
            ),
        ],
        [
            sg.Text(
                "learningRate",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["constant", "invscaling", "adaptive"],
                key="learningRate",
                default_value=utils.mlpParam["learningRate"],
                readonly=True,
                size=(8, 6),
            ),
            sg.Text(
                "learningRateInit",
                size=(11, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.mlpParam["learningRateInit"],
                key="learningRateInit",
                tooltip="float > 0",
                size=(10, 1),
            ),
            sg.Text(
                "powerT",
                size=(5, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.mlpParam["powerT"],
                key="powerT",
                tooltip="int >= 0",
                size=(10, 1),
            ),
            sg.Text(
                "maxIter",
                size=(6, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.mlpParam["maxIter"],
                key="maxIter",
                tooltip="int > 0",
                size=(10, 1),
            ),
            sg.Checkbox(
                "shuffle",
                key="shuffle",
                enable_events=True,
                default=utils.mlpParam["shuffle"],
                size=(7, 1),
            ),
        ],
        [
            sg.Text(
                "randomState",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.mlpParam["randomState"],
                key="randomState",
                tooltip="None or int >= 0",
                size=(10, 1),
            ),
            sg.Checkbox(
                "warmStart",
                key="warmStart",
                enable_events=True,
                default=utils.mlpParam["warmStart"],
                size=(7, 1),
            ),
            sg.Text(
                "momentum",
                size=(8, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.mlpParam["momentum"],
                key="momentum",
                tooltip="0 < float < 1",
                size=(10, 1),
            ),
            sg.Checkbox(
                "nesterovsMomentum",
                key="nesterovsMomentum",
                enable_events=True,
                default=utils.mlpParam["nesterovsMomentum"],
                size=(10, 1),
            ),
            sg.Checkbox(
                "earlyStopping",
                key="earlyStopping",
                enable_events=True,
                default=utils.mlpParam["earlyStopping"],
                size=(10, 1),
            ),
        ],
        [
            sg.Text(
                "validationFraction",
                size=(13, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.mlpParam["validationFraction"],
                key="validationFraction",
                tooltip="0 < float < 1",
                size=(10, 1),
            ),
            sg.Text(
                "beta1",
                size=(4, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.mlpParam["beta1"],
                key="beta1",
                tooltip="0 <= float < 1",
                size=(10, 1),
            ),
            sg.Text(
                "beta2",
                size=(4, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.mlpParam["beta2"],
                key="beta2",
                tooltip="0 <= float < 1",
                size=(10, 1),
            ),
            sg.Text(
                "nIterNoChange",
                size=(11, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.mlpParam["nIterNoChange"],
                key="nIterNoChange",
                tooltip="int > 0",
                size=(10, 1),
            ),
            sg.Text(
                "maxFun",
                size=(6, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.mlpParam["maxFun"],
                key="maxFun",
                tooltip="int > 0",
                size=(10, 1),
            ),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Button("默认", key="default", enable_events=True),
        ],
    ]


def __createXGBoostParamLayout():
    return [
        [
            sg.Text(
                "maxDepth",
                size=(7, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.xgboostParam["maxDepth"],
                key="maxDepth",
                tooltip="None or int > 0",
                size=(10, 1),
            ),
            sg.Text(
                "maxLeaves",
                size=(8, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.xgboostParam["maxLeaves"],
                key="maxLeaves",
                tooltip="None or int >= 0",
                size=(10, 1),
            ),
            sg.Text(
                "maxBin",
                size=(6, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.xgboostParam["maxBin"],
                key="maxBin",
                tooltip="None or int > 0",
                size=(10, 1),
            ),
            sg.Text(
                "growPolicy",
                size=(8, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["", "0", "1"],
                key="growPolicy",
                default_value=utils.xgboostParam["growPolicy"],
                tooltip=
                "0: favor splitting at nodes closest to the node, i.e. grow depth-wise. 1: favor splitting at nodes with highest loss change.",
                readonly=True,
                size=(8, 6),
            ),
        ],
        [
            sg.Text(
                "learningRate",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.xgboostParam["learningRate"],
                key="learningRate",
                tooltip="None or 0 <= float <= 1",
                size=(10, 1),
            ),
            sg.Text(
                "nEstimators",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.xgboostParam["nEstimators"],
                key="nEstimators",
                tooltip="int > 0",
                size=(10, 1),
            ),
            sg.Text(
                "objective",
                size=(6, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                [
                    "binary:logistic",
                    "binary:logitraw",
                    "multi:softmax",
                    "multi:softprob",
                ],
                key="objective",
                default_value=utils.xgboostParam["objective"],
                readonly=True,
                size=(8, 6),
            ),
            sg.Text(
                "booster",
                size=(6, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["", "gbtree", "gblinear", "dart"],
                key="booster",
                default_value=utils.xgboostParam["booster"],
                readonly=True,
                size=(8, 6),
            ),
        ],
        [
            sg.Text(
                "treeMethod",
                size=(8, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["", "auto", "hist", "gpu_hist"],
                key="treeMethod",
                default_value=utils.xgboostParam["treeMethod"],
                readonly=True,
                size=(8, 6),
            ),
            sg.Text(
                "gamma",
                size=(5, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.xgboostParam["gamma"],
                key="gamma",
                tooltip="None or float >= 0",
                size=(10, 1),
            ),
            sg.Text(
                "minChildWeight",
                size=(11, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.xgboostParam["minChildWeight"],
                key="minChildWeight",
                tooltip="None or float >= 0",
                size=(10, 1),
            ),
            sg.Text(
                "maxDeltaStep",
                size=(10, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.xgboostParam["maxDeltaStep"],
                key="maxDeltaStep",
                tooltip="None or float >= 0",
                size=(10, 1),
            ),
        ],
        [
            sg.Text(
                "subsample",
                size=(8, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.xgboostParam["subsample"],
                key="subsample",
                tooltip="None or 0 < float <= 1",
                size=(10, 1),
            ),
            sg.Text(
                "samplingMethod",
                size=(12, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["", "uniform", "gradient_based"],
                key="samplingMethod",
                default_value=utils.xgboostParam["samplingMethod"],
                readonly=True,
                size=(8, 6),
            ),
            sg.Text(
                "colsampleBytree",
                size=(12, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.xgboostParam["colsampleBytree"],
                key="colsampleBytree",
                tooltip="None or 0 < float <= 1",
                size=(10, 1),
            ),
            sg.Text(
                "colsampleBylevel",
                size=(13, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.xgboostParam["colsampleBylevel"],
                key="colsampleBylevel",
                tooltip="None or 0 < float <= 1",
                size=(10, 1),
            ),
        ],
        [
            sg.Text(
                "colsampleBynode",
                size=(13, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.xgboostParam["colsampleBynode"],
                key="colsampleBynode",
                tooltip="None or 0 < float <= 1",
                size=(10, 1),
            ),
            sg.Text(
                "scalePosWeight",
                size=(12, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.xgboostParam["scalePosWeight"],
                key="scalePosWeight",
                tooltip="None or int > 0",
                size=(10, 1),
            ),
            sg.Text(
                "baseScore",
                size=(8, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.xgboostParam["baseScore"],
                key="baseScore",
                tooltip="None or 0 <= float <= 1",
                size=(10, 1),
            ),
            sg.Text(
                "randomState",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.xgboostParam["randomState"],
                key="randomState",
                tooltip="None or int >= 0",
                size=(10, 1),
            ),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Button("默认", key="default", enable_events=True),
        ],
    ]


def __createAdaBoostParamLayout():
    return [
        [
            sg.Text(
                "DecisionTree",
                size=(35, 1),
                font=("Helvetica", 12),
                text_color="yellow",
            )
        ],
        [
            sg.Text(
                "criterion",
                size=(6, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["gini", "entropy", "log_loss"],
                key="criterion",
                default_value=utils.adaboostParam["criterion"],
                readonly=True,
                size=(8, 6),
            ),
            sg.Text(
                "splitter",
                size=(5, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["best", "random", "log_loss"],
                key="splitter",
                default_value=utils.adaboostParam["splitter"],
                readonly=True,
                size=(8, 6),
            ),
            sg.Text(
                "maxDepth",
                size=(7, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.adaboostParam["maxDepth"],
                key="maxDepth",
                tooltip="None or int > 0",
                size=(10, 1),
            ),
            sg.Text(
                "minSamplesSplit",
                size=(12, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.adaboostParam["minSamplesSplit"],
                key="minSamplesSplit",
                tooltip="int > 0 or 0 < float <= 1",
                size=(10, 1),
            ),
        ],
        [
            sg.Text(
                "minSamplesLeaf",
                size=(12, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.adaboostParam["minSamplesLeaf"],
                key="minSamplesLeaf",
                tooltip="int > 0 or 0 < float <= 1",
                size=(10, 1),
            ),
            sg.Text(
                "minWeightFractionLeaf",
                size=(17, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.adaboostParam["minWeightFractionLeaf"],
                key="minWeightFractionLeaf",
                tooltip="0 <= float <= 1",
                size=(10, 1),
            ),
            sg.Text(
                "maxFeatures",
                size=(10, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.adaboostParam["maxFeatures"],
                key="maxFeatures",
                tooltip='None or "sqrt" or "log2" or int > 0 or 0 < float <= 1',
                size=(10, 1),
            ),
            sg.Text(
                "randomState",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.adaboostParam["dtRandomState"],
                key="dtRandomState",
                tooltip="None or int >= 0",
                size=(10, 1),
            ),
        ],
        [
            sg.Text(
                "maxLeafNodes",
                size=(11, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.adaboostParam["maxLeafNodes"],
                key="maxLeafNodes",
                tooltip="None or int > 0",
                size=(10, 1),
            ),
            sg.Text(
                "minImpurityDecrease",
                size=(15, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.adaboostParam["minImpurityDecrease"],
                key="minImpurityDecrease",
                tooltip="float >= 0",
                size=(10, 1),
            ),
            sg.Text(
                "classWeight",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["", "balanced"],
                key="classWeight",
                default_value=utils.adaboostParam["classWeight"],
                readonly=True,
                size=(8, 6),
            ),
            sg.Text(
                "ccpAlpha",
                size=(7, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.adaboostParam["ccpAlpha"],
                key="ccpAlpha",
                tooltip="float >= 0",
                size=(10, 1),
            ),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Text(
                "AdaBoost",
                size=(35, 1),
                font=("Helvetica", 12),
                text_color="yellow",
            )
        ],
        [
            sg.Text(
                "nEstimators",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.adaboostParam["nEstimators"],
                key="nEstimators",
                tooltip="int > 0",
                size=(10, 1),
            ),
            sg.Text(
                "learningRate",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.adaboostParam["learningRate"],
                key="learningRate",
                tooltip="float > 0",
                size=(10, 1),
            ),
            sg.Text(
                "algorithm",
                size=(7, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputCombo(
                ["SAMME", "SAMME.R"],
                key="algorithm",
                default_value=utils.adaboostParam["algorithm"],
                readonly=True,
                size=(10, 6),
            ),
            sg.Text(
                "randomState",
                size=(9, 1),
                auto_size_text=False,
                justification="left",
            ),
            sg.InputText(
                utils.adaboostParam["abRandomState"],
                key="abRandomState",
                tooltip="None or int >= 0",
                size=(10, 1),
            ),
        ],
        [sg.Text("_" * 80)],
        [
            sg.Button("默认", key="default", enable_events=True),
        ],
    ]


def createRandomForestParamWindow():
    layout = __createRandomForestParamLayout() + __createBottomLayout()
    return sg.Window("随机森林参数设置", layout, modal=True)


def createSVMParamWindow():
    layout = __createSVMParamLayout() + __createBottomLayout()
    return sg.Window("SVM参数设置", layout, modal=True)


def createMLPParamWindow():
    layout = __createMLPParamLayout() + __createBottomLayout()
    return sg.Window("MLP参数设置", layout, modal=True)


def createXGBoostParamWindow():
    layout = __createXGBoostParamLayout() + __createBottomLayout()
    return sg.Window("XGBoost参数设置", layout, modal=True)


def createAdaBoostParamWindow():
    layout = __createAdaBoostParamLayout() + __createBottomLayout()
    return sg.Window("AdaBoost参数设置", layout, modal=True)


def showFeaturesDialog(headings: list, values: list):
    layout = [
        [
            sg.Table(
                values,
                headings=headings,
                key="featuresTable",
                vertical_scroll_only=False,
                num_rows=min(len(values), 30),
                alternating_row_color="#626366",
                display_row_numbers=True,
            )
        ],
        [sg.Text("_" * 80)],
        [
            sg.Button("关闭", key="close", enable_events=True),
        ],
    ]
    window = sg.Window("特征", layout, resizable=True, modal=True)
    while True:
        event, vals = window.read(timeout=500)
        if event in (None, "Exit", "close"):
            break
    window.Close()
