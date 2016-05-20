import warnings
warnings.warn("boosted_trees.py has been deprecated in favor of LensClassifierExperiment.py.")


import sklearn.metrics
import sklearn.ensemble
import ggplot as gg
import pandas as pd
from AstroImageMunger import *

swmunge = AstroImageMunger() 
# recover features image by image in a pd dataframe
n_train = int(round(11512 * 9 / 10))
n_test  = int(round(11512 / 10))
NUM_FEATURES = 43
features_train = np.zeros((n_train, NUM_FEATURES))
features_test  = np.zeros((n_test, NUM_FEATURES))
labels_train = np.zeros(n_train)
labels_test  = np.zeros(n_test)
for i in range(n_train):
    features_train[i, :], labels_train[i] = swmunge.nextExample(datum_type="sextractor", CV_type="train")
for i in range(n_test):
    features_test[i, :],  labels_test[i]  = swmunge.nextExample(datum_type="sextractor", CV_type="test")

# =====   testing different max depths   ======

fprs = []
tprs = []
depths = []
for depth in [2, 3, 4, 5]:
    print "Working on depth " + str(depth)
    #train, test
    trees = sklearn.ensemble.GradientBoostingClassifier(max_depth=depth)
    trees.fit(X=features_train, y=labels_train)
    predictions_test = trees.predict_proba(X=features_test)
    fpr, tpr, thresh = sklearn.metrics.roc_curve(y_true = labels_test, y_score = predictions_test[:, 1])
    fprs.extend(fpr)
    tprs.extend(tpr)
    depths.extend(depth * np.ones(len(tpr)))

#plot, save
depths = [str(d) for d in depths] #turn this to string for categorical colour scheme
to_plot = pd.DataFrame({"FPR": fprs, "TPR": tprs, "depth": depths})
p = gg.ggplot(data = to_plot, aesthetics = gg.aes(x = "FPR", y = "TPR", colour = "depth")) + \
    gg.geom_line() + gg.ggtitle("Tree ROCs by max depth") + \
    gg.xlab("FPR") + gg.ylab("TPR")
gg.ggsave(filename="results/tree_rocs_by_depth.png", plot = p)


# =====   testing different numbers of trees   ======
# This test takes just long enough to be irritating.
fprs = []
tprs = []
n_trees_column = []
for n_trees in [100, 200, 300, 400, 500]:
    print "Working on n_trees " + str(n_trees)
    #train, test
    trees = sklearn.ensemble.GradientBoostingClassifier(max_depth=5, n_estimators = n_trees)
    trees.fit(X=features_train, y=labels_train)
    predictions_test = trees.predict_proba(X=features_test)
    fpr, tpr, thresh = sklearn.metrics.roc_curve(y_true = labels_test, y_score = predictions_test[:, 1])
    fprs.extend(fpr)
    tprs.extend(tpr)
    n_trees_column.extend(n_trees * np.ones(len(tpr)))

#plot, save
n_trees_column = [str(d) for d in n_trees_column] #turn this to string for categorical colour scheme
to_plot = pd.DataFrame({"FPR": fprs, "TPR": tprs, "n_trees": n_trees_column})
p = gg.ggplot(data = to_plot, aesthetics = gg.aes(x = "FPR", y = "TPR", colour = "n_trees")) + \
    gg.geom_line() + gg.ggtitle("Tree ROCs by number of trees") + \
    gg.xlab("FPR") + gg.ylab("TPR")
gg.ggsave(filename="results/tree_rocs_by_n_trees.png", plot = p)
