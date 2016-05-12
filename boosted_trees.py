import sklearn.metrics
import sklearn.ensemble
import matplotlib.pyplot as plt
from AstroImageMunger import *

swmunge = AstroImageMunger() 
# recover features image by image in a pd dataframe
n_train = 5#int(round(11512 * 9 / 10))
n_test  = 5#int(round(11512 / 10))
features_train = np.zeros((n_train, 43))
features_test  = np.zeros((n_test, 43))
labels_train = np.zeros(n_train)
labels_test  = np.zeros(n_test)
for i in range(n_train):
    features_train[i, :], labels_train[i] = swmunge.nextExample(datum_type="sextractor", CV_type="train")
for i in range(n_test):
    features_test[i, :],  labels_test[i]  = swmunge.nextExample(datum_type="sextractor", CV_type="train")

for depth in [2, 3, 4, 5]:
    #train, test
    trees = sklearn.ensemble.GradientBoostingClassifier(max_depth=depth)
    trees.fit(X=features_train, y=labels_train)
    predictions_test = trees.predict_proba(X=features_test)
    fpr, tpr, thresh = sklearn.metrics.roc_curve(y_true = labels_test, y_score = predictions_test[:, 0])
    #plot, save
    figname = "TPR versus FPR for max_depth " + str(depth)
    filename = "roc_max_depth_" + str(depth)
    plt.clf()
    plt.plot(fpr, tpr)
    plt.title(figname)
    plt.savefig("results/" + figname + ".png")
