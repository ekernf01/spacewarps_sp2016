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
        self.batch_size = 200
        self.n_distinct_batches = int(np.ceil(self.n_train / float(self.batch_size)))
        self.debug_mode_string = ""

        if self.debug_mode:
            self.n_train = self.n_train / 20
            self.n_test = self.n_test / 20
            self.batch_size = 20
            self.n_distinct_batches = 1
            self.debug_mode_string = "_DEBUG_MODE"

        self.static_batch = self.swmunge.get_batch(self.batch_size, CV_type="train")
        self.labels_train = np.zeros(self.n_train)
        self.labels_test  = np.zeros(self.n_test)
        self.valid_exp_names = ['lenet|nkern|lambda',
                                'trees|1obj|max_depth', 'trees|1obj|num_trees',
                                'trees|3obj|max_depth', 'trees|3obj|num_trees']
        self.suggested_parvals = [[0.001, 0.01, 0.1, 1],
                                  [2,3,4,5], [50, 100, 250],
                                  [2,3,4,5], [50, 100, 250]]

    def init_features(self, experiment_type):
        if experiment_type in ['trees|1obj|max_depth', 'trees|1obj|num_trees']:
            NUM_FEATURES = self.swmunge.SEXTRACTOR_FT_OUT
            datum_type = "sextractor|1obj"
        elif experiment_type in ['trees|3obj|max_depth', 'trees|3obj|num_trees']:
            NUM_FEATURES = self.swmunge.SEXTRACTOR_FT_OUT * 3 + 6
            datum_type = "sextractor|3obj"
        elif experiment_type == 'lenet|nkern':
            return
        else:
            raise Exception("Invalid experiment type or out-of-date code in init_features")

        self.features_train = np.zeros((self.n_train, NUM_FEATURES))
        self.features_test  = np.zeros((self.n_test,  NUM_FEATURES))
        for i in range(self.n_train):
            self.features_train[i, :], self.labels_train[i] = self.swmunge.nextExample(datum_type, CV_type="train")
        for i in range(self.n_test):
            self.features_test[i, :],  self.labels_test[i]  = self.swmunge.nextExample(datum_type, CV_type="test")
        return

    def get_training_batch(self, batch_size):
        #if self.debug_mode:
        #    return self.static_batch
        return self.swmunge.get_batch(batch_size, CV_type="train")

    def run(self, experiment_type, parvals):
        self.init_features(experiment_type)

        fprs = []
        tprs = []
        parvals_long = []
        for par in parvals:
            print "Working on " + experiment_type + " = " + str(par)
            #train, test
            if experiment_type in ("trees|1obj|max_depth", "trees|3obj|max_depth"):
                model = sklearn.ensemble.GradientBoostingClassifier(max_depth=par)
                model.fit(X=self.features_train, y=self.labels_train)
            elif experiment_type in ("trees|1obj|num_trees, trees|3obj|num_trees"):
                model = sklearn.ensemble.GradientBoostingClassifier(max_depth=5, n_estimators=par)
                model.fit(X=self.features_train, y=self.labels_train)
            elif experiment_type == "lenet|nkern|lambda":
                model = TheanoCNN.LeNet(image_size = list(self.swmunge.image_shape), nkerns = [10, 10], lambduh = par,
                                        get_training_batch = self.get_training_batch, batch_size=self.batch_size)
                num_passes = 4
                model.fit(self.n_distinct_batches * num_passes)
                net_path = "results/saved_net_lam=" + str(par) + self.debug_mode_string + ".pkl"
                model.save(net_path)
                model = TheanoCNN.LeNet(path = net_path)
                self.features_test, self.labels_test = self.swmunge.get_batch(self.n_test, CV_type="test")
            elif experiment_type in self.valid_exp_names:
                raise Exception("Oops! Programming error! self.valid_exp_names doesn't match this block of conditionals.")
            else:
                print experiment_type
                raise Exception("experiment_type not valid. Must be one of " + ", ".join(self.valid_exp_names))

            preds = model.predict_proba(self.features_test)
            fpr, tpr, thresh = sklearn.metrics.roc_curve(y_true = self.labels_test, y_score = preds[:, 1])
            fprs.extend(fpr)
            tprs.extend(tpr)
            parvals_long.extend(par * np.ones(len(tpr)))
        return experiment_type, pd.DataFrame({"FPR": fprs, "TPR": tprs, "parameter": parvals_long}), preds

    def plot_roc(self, experiment_type, to_plot):
        # turn this to string for categorical colour scheme
        to_plot.loc[:, "parameter"] = [str(par) for par in to_plot.loc[:, "parameter"]]
        p = gg.ggplot(data = to_plot, aesthetics = gg.aes(x = "FPR", y = "TPR", colour = "parameter")) + \
            gg.geom_line(gg.aes(x = "FPR", y = "TPR", colour = "parameter")) + \
            gg.ggtitle(experiment_type) + gg.xlab("FPR") + gg.ylab("TPR")
        gg.ggsave(filename = "results/" + experiment_type + self.debug_mode_string + ".png", plot = p)
        return
