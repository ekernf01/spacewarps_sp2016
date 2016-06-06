import sklearn.metrics
import sklearn.ensemble
import ggplot as gg
import matplotlib.pyplot as plt
import pandas as pd
from AstroImageMunger import *
import TheanoCNN


class LensClassifierExperiment():

    def __init__(self, mode = "full_run"):
        self.swmunge = AstroImageMunger()
        self.n_train = int(round(11512 * 9 / 10))
        self.n_test  = int(round(11512 / 10))
        self.mode = mode
        self.batch_size = 200
        self.n_distinct_batches = int(np.ceil(self.n_train / float(self.batch_size)))

        if self.mode == "debug":
            self.n_train = self.n_train / 20
            self.n_test = self.n_test / 20
            self.batch_size = 5
            self.n_distinct_batches = 100

        if self.mode == "dry_run":
            self.n_train = self.n_train / 4
            self.n_test = self.n_test / 4
            self.batch_size = 5
            self.n_distinct_batches = 2000

        self.static_batch = self.swmunge.get_batch(self.batch_size, CV_type="train")
        self.labels_train = np.zeros(self.n_train)
        self.labels_test  = np.zeros(self.n_test)
        self.valid_exp_names = ['lenet|lambda',
                                'lenet|nkern',
                                'lenet|npass',
                                'trees|1obj|max_depth', 'trees|1obj|num_trees',
                                'trees|3obj|max_depth', 'trees|3obj|num_trees']
        self.suggested_parvals = [[10 ** -13, 10 ** -11, 10 ** -9, 10 ** -7, 10 ** -5, 10 ** -3],
                                  [1, 5, 10],
                                  [1, 4, 10],
                                  [2,3,4,5], [50, 100, 250],
                                  [2,3,4,5], [50, 100, 250]]

    def init_features(self, experiment_type):
        if experiment_type in ['trees|1obj|max_depth', 'trees|1obj|num_trees']:
            NUM_FEATURES = self.swmunge.SEXTRACTOR_FT_OUT
            datum_type = "sextractor|1obj"
        elif experiment_type in ['trees|3obj|max_depth', 'trees|3obj|num_trees']:
            NUM_FEATURES = self.swmunge.SEXTRACTOR_FT_OUT * 3 + 6
            datum_type = "sextractor|3obj"
        elif experiment_type[0:6] == 'lenet|':
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
        if self.mode == "debug":
            return self.static_batch
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
            elif experiment_type[0:6] == "lenet|":
                if experiment_type[6:12] == "lambda":
                    nkerns = [2, 2]
                    lambduh = par
                    num_passes = 4
                elif experiment_type[6:11] == "nkern":
                    nkerns = [par, par]
                    lambduh = 10 ** -7
                    num_passes = 4
                elif experiment_type[6:11] == "npass":
                    nkerns = [5, 5]
                    lambduh = 10 ** -7
                    num_passes = par
                else:
                    raise Exception("Oops! Programming error! self.valid_exp_names doesn't match this block of conditionals.")

                model = TheanoCNN.LeNet(image_size = list(self.swmunge.image_shape), nkerns = nkerns, lambduh = lambduh,
                                        get_training_batch = self.get_training_batch, batch_size=self.batch_size,
                                        mode = self.mode)
                (cum_costs, cum_errs, cum_penalties) = model.fit(self.n_distinct_batches * num_passes)
                net_path = "results/saved_net/" + experiment_type + "=" + str(par) + "_mode=" + self.mode + ".pkl"
                model.save(net_path)
                fig_path = "results/training_progress/" + experiment_type + "=" + str(par) + "_mode=" + self.mode + ".png"
                plt.clf()
                plt.plot(cum_costs     / range(len(cum_costs)), "b+")
                plt.plot(cum_errs      / range(len(cum_errs)), "rx")
                plt.plot(cum_penalties / range(len(cum_penalties)), "gx")
                plt.plot(cum_costs[1:] - cum_costs[:-1], "ko")
                assert all(x < 0.00000001 for x in cum_costs - cum_errs - cum_penalties)
                plt.legend(labels = ["cost", "error", "penalty", "non-cumulative cost"])
                plt.title("training_progress")
                plt.savefig(fig_path)
                model = TheanoCNN.LeNet(path = net_path)
                self.features_test, self.labels_test = self.swmunge.get_batch(self.n_test, CV_type="test")
            elif experiment_type in self.valid_exp_names:
                raise Exception("Oops! Programming error! self.valid_exp_names doesn't match this block of conditionals.")
            else:
                print experiment_type
                raise Exception("experiment_type not valid. Must be one of " + ", ".join(self.valid_exp_names))

            preds = model.predict_proba(self.features_test)
            print "Test proportion: " + str(np.mean(self.labels_test))
            print "preds mean: " + str(np.mean(preds[:, 1]))
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
        gg.ggsave(filename = "results/" + experiment_type + "_" + self.mode + ".png", plot = p)
        return
