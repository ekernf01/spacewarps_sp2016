import sklearn.ensemble as ensemble
from AstroImageMunger import *

swmunge = AstroImageMunger(test_mode = True) #switch this out of test mode eventually
# recover features in big fat pd dataframe
features = swmunge.loadFeatures()

trees = ensemble.GradientBoostingClassifier()