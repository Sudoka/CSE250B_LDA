"""
Figure out organization of the given .mat data file.
"""

import scipy.io as sio

data = sio.loadmat('classic400.mat')

# get keys
print data.keys()

# words in vocabulary
print data['classicwordlist']

# counts of each word
print data['truelabels']

#print data['classic400']
