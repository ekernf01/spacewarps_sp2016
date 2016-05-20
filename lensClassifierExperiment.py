import sklearn.metrics
import sklearn.ensemble
import ggplot as gg
import pandas as pd
from AstroImageMunger import *
import TheanoCNN


class LensClassifierExperiment():

    def __init__(self, debug_mode = False):
        self.swmunge = AstroImageMunger()
        self.n_train = int(round(11512 * 9 / 10))
        self.n_test  = int(round(11512 / 10))
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.n_train = self.n_train / 100
            self.n_test = self.n_test / 100

        self.batch_size = 200
        self.n_batches = int(np.ceil(self.n_train / 200.0))
        self.NUM_FEATURES = 43
        self.features_train = np.zeros((self.n_train, self.NUM_FEATURES))
        self.features_test  = np.zeros((self.n_test, self.NUM_FEATURES))
        self.labels_train = np.zeros(self.n_train)
        self.labels_test  = np.zeros(self.n_test)
        self.valid_exp_names = ['trees|max_depth', 'trees|num_trees', 'lenet|nkern']
        self.suggested_parvals = [[2,3,4,5], [50, 100, 250], [20]]
        for i in range(self.n_train):
            self.features_train[i, :], self.labels_train[i] = self.swmunge.nextExample(datum_type="sextractor", CV_type="train")
        for i in range(self.n_test):
            self.features_test[i, :],  self.labels_test[i]  = self.swmunge.nextExample(datum_type="sextractor", CV_type="test")

    def get_training_batch(self, batch_size):
        return self.swmunge.get_batch(batch_size, CV_type="train")

    def run(self, experiment_type, parvals):
        fprs = []
        tprs = []
        parvals_long = []
        for par in parvals:
            print "Working on " + experiment_type + " = " + str(par)
            #train, test
            if experiment_type == "trees|max_depth":
                model = sklearn.ensemble.GradientBoostingClassifier(max_depth=par)
                model.fit(X=self.features_train, y=self.labels_train)
            elif experiment_type == "trees|num_trees":
                model = sklearn.ensemble.GradientBoostingClassifier(max_depth=5, n_estimators=par)
                model.fit(X=self.features_train, y=self.labels_train)
            elif experiment_type == "lenet|nkern":
                model = TheanoCNN.LeNet(image_size = list(self.swmunge.image_shape), nkerns = [par, par],
                                        get_training_batch = self.get_training_batch, batch_size=200)
                model.fit(self.n_batches)
                self.features_test, self.labels_test = self.swmunge.get_batch(self.n_test, CV_type="test")
            else:
                raise Exception("experiment_type not valid. Must be one of " + ", ".join(self.valid_exp_names))

            predictions_test = model.predict_proba(self.features_test)
            fpr, tpr, thresh = sklearn.metrics.roc_curve(y_true = self.labels_test, y_score = predictions_test[:, 1])
            fprs.extend(fpr)
            tprs.extend(tpr)
            parvals_long.extend(par * np.ones(len(tpr)))
        return experiment_type, pd.DataFrame({"FPR": fprs, "TPR": tprs, "parameter": parvals_long})

    def plot_roc(self, experiment_type, to_plot):
        # turn this to string for categorical colour scheme
        to_plot.loc[:, "parameter"] = [str(par) for par in to_plot.loc[:, "parameter"]]
        p = gg.ggplot(data = to_plot, aesthetics = gg.aes(x = "FPR", y = "TPR", colour = "parameter")) + \
            gg.geom_line(gg.aes(x = "FPR", y = "TPR", colour = "parameter")) + \
            gg.ggtitle(experiment_type) + gg.xlab("FPR") + gg.ylab("TPR")
        if self.debug_mode:
            debug_indicator = "_DEBUG_MODE"
        else:
            debug_indicator = ""
        gg.ggsave(filename = "results/" + experiment_type + debug_indicator + ".png", plot = p)
        return
