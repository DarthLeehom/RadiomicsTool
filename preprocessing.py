import ui

__preprocessTypes = {
    "useSquare": False,
    "useSquareRoot": False,
    "useLogarithm": False,
    "useExponential": False,
    "useLoG": False,
    "sigma": {"sigma": []},
    "useWavelet": False,
    "waveletParam": {"wavelet": "", "start_level": 0, "level": 1}
    # "startLevel": int, 0 based level of wavelet which should be used as first set of decompositions
    # "level": int, number of levels of wavelet decompositions from which a signature is calculated
}


def checkPreprocessingType(values:dict):
    for str in values["sigma"].split(","):
        if str.isdigit() and int(str) != 0:
            __preprocessTypes["sigma"]["sigma"].append(int(str))
        else:
            ui.popupError("Sigma error, please check your input!")
            return False, __preprocessTypes
    if not values["startLevel"].isdigit() or not values["level"].isdigit():
        ui.popupError("startLevel or level error, please check your input!")
        return False, __preprocessTypes
    else:
        __preprocessTypes["waveletParam"]["start_level"] = int(values["startLevel"])
        __preprocessTypes["waveletParam"]["level"] = int(values["level"])

    __preprocessTypes["useSquare"] = values["square"]
    __preprocessTypes["useSquareRoot"] = values["squareRoot"]
    __preprocessTypes["useLogarithm"] = values["logarithm"]
    __preprocessTypes["useExponential"] = values["exponential"]
    __preprocessTypes["useLoG"] = values["LoG"]
    __preprocessTypes["useWavelet"] = values["wavelet"]
    __preprocessTypes["waveletParam"]["wavelet"] = values["waveletType"]

    return True, __preprocessTypes
