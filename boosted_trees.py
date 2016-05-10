import sklearn.ensemble as ensemble
from AstroImageMunger import *

swmunge = AstroImageMunger() 
# recover features image by image in a pd dataframe
feature = swmunge.nextExample(datum_type="sextractor")
print feature
trees = ensemble.GradientBoostingClassifier()