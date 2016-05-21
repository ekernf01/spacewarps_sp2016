# Run this once at the start of the project to pre-process data.

from AstroImageMunger import *

swmunge = AstroImageMunger(test_mode = False)

# Make FITS files. If any already exist, it will not harm them... by default.
#swmunge.pngsToFits()

# Extract and save features.  If any already exist, they will be overwritten by default.
swmunge.saveFeatures()

