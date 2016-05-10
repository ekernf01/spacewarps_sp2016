import TheanoCNN
import numpy as np
from AstroImageMunger import AstroImageMunger
swmunge = AstroImageMunger()
def get_batch(batch_size, CV_type):
    return swmunge.get_batch(batch_size, CV_type, datum_type = "image")

sw_net = TheanoCNN.LeNet()
sw_net.eval_lenet5(get_batch, list(swmunge.image_shape), n_epochs=2, batch_size=5)