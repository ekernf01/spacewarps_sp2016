
# coding: utf-8

# ### Make FITS files
# 
# Currently broken, I suspect. When I run sextractor directly on a FITS file from another source, it gives me a better error than it does on my homemade FITS files.

# In[1]:

from AstroImageMunger import *
swmunge = AstroImageMunger()
swmunge.convert_pngs_to_fits(test_mode = True) #switch this out of test mode when you have time


# In[ ]:

print swmunge.get_features()[0]["table"]

