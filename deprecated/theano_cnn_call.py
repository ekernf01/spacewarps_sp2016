
import warnings
warnings.warn("theano_cnn_call.py has been deprecated in favor of LensClassifierExperiment.py.")


import TheanoCNN
from six.moves import cPickle
from time import gmtime, strftime
metadata = strftime("%Y-%m-%d %H:%M:%S", gmtime())


from AstroImageMunger import AstroImageMunger
swmunge = AstroImageMunger()
def get_batch(batch_size, CV_type):
    return swmunge.get_batch(batch_size, CV_type = "train", datum_type = "image")

sw_net = TheanoCNN.LeNet(image_size = list(swmunge.image_shape), get_batch = get_batch, batch_size=5)
sw_net.fit(n_batches=20)
test_probs = sw_net.predict_proba()

mypath = "results/trained_net_" + metadata + ".pkl"
print "Saving results to " + mypath
with open(mypath, "wb") as f:
    cPickle.dump(obj = {"predictions":test_probs,
                        "trained_net":sw_net},
                 file = f,
                 protocol=cPickle.HIGHEST_PROTOCOL)

