import utils
import ui


def showFeatureConfigWindow(featureType: str, enabledFeatures: list):
    featureConfigWindow = ui.createFeatureConfigWindow(
        featureType,
        enabledFeatures,
        "{} Feature Config".format(featureType.capitalize()),
    )
    if featureConfigWindow is None:
        return None

    while True:
        event, values = featureConfigWindow.read(timeout=500)
        if event in (None, "Exit", "cancel"):
            break
        if event == "all":
            featureNames = utils.getAllFeatureNames()[featureType]
            for name in featureNames:
                featureConfigWindow[name].update(
                    True if values[event] else False)
        if event == "ok":
            featureNames = utils.getAllFeatureNames()[featureType]
            for name in featureNames:
                if values[name]:
                    if name not in enabledFeatures:
                        enabledFeatures.append(name)
                else:
                    if name in enabledFeatures:
                        enabledFeatures.remove(name)
            break

    featureConfigWindow.Close()
    return enabledFeatures
