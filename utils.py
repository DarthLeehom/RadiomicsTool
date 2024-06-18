ttestParam = {
    "leveneCenter": "median",  # "mean", "median", "trimmed"
    "leveneProportiontocut": 0.05,  # float, valid range is [0, 1]
    "ttestAxis": 0,  # None or int >= 0
    "ttestNanPolicy": "propagate",  # "propagate", "raise", "omit"
    "ttestPermutations": None,  # None or int >= 0
    "ttestRandomState": None,  # None or int >= 0
    "ttestAlternative": "two-sided",  # "two-sided", "less", "greater"
    "ttestTrim": 0,  # float, valid range is [0, 0.5)
}

lassoParam = {
    "customAlphas": False,
    "start": None,  # num(both negative and positive int)
    "stop": None,  # num(both negative and positive int), stop > start
    "num": None,  # int > 0
    "base": 10,  # int > 0
    "eps": 0.001,  # positive float, ``eps=1e-3`` means that "alpha_min / alpha_max = 1e-3"
    "nAlphas": 100,  # int > 0
    "fitIntercept": True,  # bool
    "maxIter": 1000,  # int > 0
    "cv": None,  # None or int > 0, Determines the cross-validation splitting strategy. For int/None inputs, :class:"KFold" is used.
    "randomState": None,  # None or int >= 0
    "selection": "cyclic",  # "cyclic", "random"
}

pcaParam = {
    "nComponents": None,  # None, int > 0, 0 < float < 1, "mle"
    "whiten": False,  # bool
    "svdSolver": "auto",  # "auto", "full", "arpack", "randomized"
    "iteratedPower": "auto",  # int >= 0 or "auto"
    "nOversamples": 10,  # int > 0
    "powerIterationNormalizer": "auto",  # "auto", "QR", "LU", "none"
    "randomState": None,  # None or int >= 0
}

muifParam = {
    "useKBest": True,  # bool
    "k": 10,  # "all" or int > 0
    "usePercentile": False,  # bool
    "percentile": 10,  # 0 < int <= 100
}

chi2Param = {
    "useKBest": True,  # bool
    "k": 10,  # "all" or int > 0
    "usePercentile": False,  # bool
    "percentile": 10,  # 0 < int <= 100
}

mRMRParam = {"method": "MIQ", "nfeats": 8}  # method: "MID", "MIQ", nfeats: int > 0

randomForestParam = {
    "nEstimators": 100,  # int > 0
    "criterion": "gini",  # "gini", "entropy", "log_loss"
    "maxDepth": None,  # None or int > 0
    "minSamplesSplit": 2,  # int > 0 or 0 < float <= 1
    "minSamplesLeaf": 1,  # int > 0 or 0 < float <= 1
    "minWeightFractionLeaf": 0,  # 0 <= float <= 1
    "maxFeatures": "sqrt",  # None or "sqrt" or "log2" or int > 0 or 0 < float <= 1
    "maxLeafNodes": None,  # None or int > 0
    "minImpurityDecrease": 0,  # float >= 0
    "bootstrap": True,  # bool
    "oobScore": False,  # bool
    "randomState": None,  # None or int >= 0
    "warmStart": False,  # bool
    "classWeight": None,  # None or "balanced" or "balanced_subsample"
    "ccpAlpha": 0,  # float >= 0
    "maxSamples": None,  # None or int > 0 or 0 < float <= 1
}

svmParam = {
    "C": 1,  # float > 0
    "kernel": "rbf",  # "linear", "poly", "rbf", "sigmoid", "precomputed"
    "degree": 3,  # int >= 0
    "gamma": "scale",  # "scale" or "auto" or float >= 0
    "coef0": 0,  # float >= 0
    "shrinking": True,  # bool
    "probability": False,  # bool
    "classWeight": None,  # None or "balanced"
    "maxIter": -1,  # -1 or int > 0
    "decisionFunctionShape": "ovr",  # "ovo", "ovr"
    "breakTies": False,  # bool
    "randomState": None,  # None or int >= 0
}

mlpParam = {
    "hiddenLayerSizes": (100,),  # (int > 0,int > 0,int > 0,int > 0,...,int > 0)
    "activation": "relu",  # "identity", "logistic", "tanh", "relu"
    "solver": "adam",  # "lbfgs", "sgd", "adam"
    "alpha": 0.0001,  # float > 0
    "batchSize": "auto",  # auto or int > 0
    "learningRate": "constant",  # "constant", "invscaling", "adaptive"
    "learningRateInit": 0.001,  # float > 0
    "powerT": 0.5,  # float >= 0
    "maxIter": 200,  # int > 0
    "shuffle": True,  # bool
    "randomState": None,  # None or int >= 0
    "warmStart": False,  # bool
    "momentum": 0.9,  # 0 < float < 1
    "nesterovsMomentum": True,  # bool
    "earlyStopping": False,  # bool
    "validationFraction": 0.1,  # 0 < float < 1
    "beta1": 0.9,  # 0 <= float < 1
    "beta2": 0.999,  # 0 <= float < 1
    "nIterNoChange": 10,  # int > 0
    "maxFun": 15000,  # int > 0
}

xgboostParam = {
    "maxDepth": None,  # None or int > 0
    "maxLeaves": None,  # None or int >= 0
    "maxBin": None,  # None or int > 0
    "growPolicy": None,  # None or 0: favor splitting at nodes closest to the node, i.e. grow depth-wise. 1: favor splitting at nodes with highest loss change.
    "learningRate": None,  # None or 0 <= float <= 1
    "nEstimators": 100,  #  int > 0
    "objective": "binary:logistic",  # "binary:logistic" or "binary:logitraw" or "multi:softmax" or "multi:softprob"
    "booster": None,  # None or "gbtree" or "gblinear" or "dart"
    "treeMethod": None,  # None or "auto" or "hist" or "gpu_hist"
    "gamma": None,  # None or float >= 0
    "minChildWeight": None,  # None or float >= 0
    "maxDeltaStep": None,  # None or float >= 0
    "subsample": None,  # None or 0 < float <= 1
    "samplingMethod": None,  # None or "uniform" or "gradient_based"
    "colsampleBytree": None,  # None or 0 < float <= 1
    "colsampleBylevel": None,  # None or 0 < float <= 1
    "colsampleBynode": None,  # None or 0 < float <= 1
    "scalePosWeight": None,  # None or int > 0
    "baseScore": None,  # None or 0 <= float <= 1
    "randomState": None,  # None or int >= 0
}

adaboostParam = {
    # -------------------DecisionTreeClassifier-------------------#
    "criterion": "gini",  # "gini", "entropy", "log_loss"
    "splitter": "best",  # "best", "random"
    "maxDepth": None,  # None or int > 0
    "minSamplesSplit": 2,  # int > 0 or 0 < float <= 1
    "minSamplesLeaf": 1,  # int > 0 or 0 < float <= 1
    "minWeightFractionLeaf": 0,  # 0 <= float <= 1
    "maxFeatures": "sqrt",  # None or "sqrt" or "log2" or int > 0 or 0 < float <= 1
    "dtRandomState": None,  # None or int >= 0
    "maxLeafNodes": None,  # None or int > 0
    "minImpurityDecrease": 0,  # float >= 0
    "classWeight": None,  # None or "balanced"
    "ccpAlpha": 0,  # float >= 0
    # -------------------DecisionTreeClassifier-------------------#
    "nEstimators": 50,  # int > 0
    "learningRate": 1,  # float > 0
    "algorithm": "SAMME.R",  # "SAMME", "SAMME.R"
    "abRandomState": None,  # None or int >= 0
}


def getWaveletTypeList():
    biorList = [
        1.1,
        1.3,
        1.5,
        2.2,
        2.4,
        2.6,
        2.8,
        3.1,
        3.3,
        3.5,
        3.7,
        3.9,
        4.4,
        5.5,
        6.8,
    ]
    coifList = [i for i in range(1, 6)]
    dbList = [i for i in range(1, 21)]
    symList = [i for i in range(2, 21)]
    waveletTypeList = (
        ["haar", "dmey"]
        + ["bior{}".format(i) for i in biorList]
        + ["rbio{}".format(i) for i in biorList]
        + ["coif{}".format(i) for i in coifList]
        + ["db{}".format(i) for i in dbList]
        + ["sym{}".format(i) for i in symList]
    )
    return waveletTypeList


def getAllFeatures():
    return {
        "shape": [
            "Elongation",
            "Flatness",
            "LeastAxisLength",
            "MajorAxisLength",
            "Maximum2DDiameterColumn",
            "Maximum2DDiameterRow",
            "Maximum2DDiameterSlice",
            "Maximum3DDiameter",
            "MeshVolume",
            "MinorAxisLength",
            "Sphericity",
            "SurfaceArea",
            "SurfaceVolumeRatio",
            "VoxelVolume",
        ],
        "firstorder": [
            "10Percentile",
            "90Percentile",
            "Energy",
            "Entropy",
            "InterquartileRange",
            "Kurtosis",
            "Maximum",
            "MeanAbsoluteDeviation",
            "Mean",
            "Median",
            "Minimum",
            "Range",
            "RobustMeanAbsoluteDeviation",
            "RootMeanSquared",
            "Skewness",
            "TotalEnergy",
            "Uniformity",
            "Variance",
        ],
        "glcm": [
            "Autocorrelation",
            "ClusterProminence",
            "ClusterShade",
            "ClusterTendency",
            "Contrast",
            "Correlation",
            "DifferenceAverage",
            "DifferenceEntropy",
            "DifferenceVariance",
            "Id",
            "Idm",
            "Idmn",
            "Idn",
            "Imc1",
            "Imc2",
            "InverseVariance",
            "JointAverage",
            "JointEnergy",
            "JointEntropy",
            "MCC",
            "MaximumProbability",
            "SumAverage",
            "SumEntropy",
            "SumSquares",
        ],
        "glrlm": [
            "GrayLevelNonUniformity",
            "GrayLevelNonUniformityNormalized",
            "GrayLevelVariance",
            "HighGrayLevelRunEmphasis",
            "LongRunEmphasis",
            "LongRunHighGrayLevelEmphasis",
            "LongRunLowGrayLevelEmphasis",
            "LowGrayLevelRunEmphasis",
            "RunEntropy",
            "RunLengthNonUniformity",
            "RunLengthNonUniformityNormalized",
            "RunPercentage",
            "RunVariance",
            "ShortRunEmphasis",
            "ShortRunHighGrayLevelEmphasis",
            "ShortRunLowGrayLevelEmphasis",
        ],
        "glszm": [
            "GrayLevelNonUniformity",
            "GrayLevelNonUniformityNormalized",
            "GrayLevelVariance",
            "HighGrayLevelZoneEmphasis",
            "LargeAreaEmphasis",
            "LargeAreaHighGrayLevelEmphasis",
            "LargeAreaLowGrayLevelEmphasis",
            "LowGrayLevelZoneEmphasis",
            "SizeZoneNonUniformity",
            "SizeZoneNonUniformityNormalized",
            "SmallAreaEmphasis",
            "SmallAreaHighGrayLevelEmphasis",
            "SmallAreaLowGrayLevelEmphasis",
            "ZoneEntropy",
            "ZonePercentage",
            "ZoneVariance",
        ],
        "gldm": [
            "DependenceEntropy",
            "DependenceNonUniformity",
            "DependenceNonUniformityNormalized",
            "DependenceVariance",
            "GrayLevelNonUniformity",
            "GrayLevelVariance",
            "HighGrayLevelEmphasis",
            "LargeDependenceEmphasis",
            "LargeDependenceHighGrayLevelEmphasis",
            "LargeDependenceLowGrayLevelEmphasis",
            "LowGrayLevelEmphasis",
            "SmallDependenceEmphasis",
            "SmallDependenceHighGrayLevelEmphasis",
            "SmallDependenceLowGrayLevelEmphasis",
        ],
        "ngtdm": [
            "Busyness",
            "Coarseness",
            "Complexity",
            "Contrast",
            "Strength",
        ],
    }


__isBusy = False


def isBusy():
    global __isBusy
    return __isBusy


def setState(isBusy:bool):
    global __isBusy
    __isBusy = isBusy
