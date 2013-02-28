import numpy as np
from scipy import sparse, spatial
import itertools
from math import log
import sys
import csv
import random
import bisect


INPUT_WORD_COUNT = 'KOS400/KOS400.csv'
INPUT_WORD_LIST = 'KOS400/KOSwordlist.csv'
OUTPUT_THIS = 'KOS400/KOS400_this.csv'
OUTPUT_THETAS = 'KOS400/KOS400_thetas.csv'
'''
INPUT_WORD_COUNT = 'classic400/classic400.csv'
INPUT_WORD_LIST = 'classic400/classicwordlist.csv'
OUTPUT_THIS = 'classic400/classic400_this.csv'
OUTPUT_THETAS = 'classic400/classic400_thetas.csv'
'''
NUM_EPOCHS = 100
MIN_DELTA = 1.0


# K is the number of topics
K = 3

# alpha and beta for LDA
alpha = 0.01
beta = 0.1

print "Loading training data ...",
sys.stdout.flush()
WC = sparse.lil_matrix(np.genfromtxt(INPUT_WORD_COUNT, delimiter=',', dtype=np.int), dtype=np.int)
print "done"
sys.stdout.flush()

#TrueLabels = np.genfromtxt('truelabels.csv', delimiter=',', dtype=np.int)

#WordList = np.genfromtxt('truelabels.csv', delimiter=',', dtype=np.int)

# M is the number of documents
# V is size of vocabulary
M, V = WC.shape

# T is the normalizing constant (sum of the sizes of all training documents)
T = WC.sum()

print "M=%d, V=%d, T=%d"%(M, V, T)


# reference each word appearance to word and document idxs
refIdxs = np.zeros((T, 2), dtype=np.int)
idxUse = 0

WC_coo = WC.tocoo()
for m,w,v in itertools.izip(WC_coo.row, WC_coo.col, WC_coo.data):
	for count in range(v):
		refIdxs[idxUse,0] = m
		refIdxs[idxUse,1] = w
		idxUse += 1
	#end for
#end for

print "idxUse=%d"%(idxUse)

# z value for each word appearance
z = np.zeros(T, dtype=np.int)

# q_wi_j is the number of times word wi occurs with topic j in the whole corpus
q = np.zeros((V,K), dtype=np.int)

# k_j is the number of words assigned to topic j in the whole corpus
k = np.zeros(K, dtype=np.int)

# n_m_j is the count of how many words within document m are assigned to topic j
n = np.zeros((M,K), dtype=np.int)

# N_m is the number of words in document m
N = np.zeros(M, dtype=np.int)

# populate q and n
for i in range(T):
	j = np.argmax(np.random.rand(K))
	z[i] = j
	m, w = refIdxs[i]
	q[w,j] += 1
	n[m,j] += 1
	k[j] += 1
	N[m] += 1
#end for

oldThi = np.negative(np.ones((V,K)))
newThi = np.negative(np.ones((V,K)))
newTheta = np.negative(np.ones((M,K)))

def computeNewThi():
	for j in range(K):
		for w in range(V):
			newThi[w,j] = log(q[w,j] + beta) - log(k[j] + (V*beta))
		#end for
	#end for
#end def

def computeNewTheta():
	for m in range(M):
		for j in range(K):
			newTheta[m,j] = log(n[m,j] + alpha) - log(N[m] + (K*alpha))
		#end for
	#end for
#end def

def getThiChange():
	tot = 0.0
	for j in range(K):
		numer = 0.0
		denom = 0.0
		for w in range(V):
			numer += (newThi[w,j] - oldThi[w,j]) ** 2
			denom += 1.0
		#end for
		tot += log(numer) - log(denom)
	#end for
	return tot / float(K)
#end def

stepDis = T / 10

delta = 10.0
numEpochs = 0
while numEpochs < NUM_EPOCHS:
	print "Computing z values ...",
	sys.stdout.flush()
	
	for i in range(T):
		if i % stepDis == 0:
			print i / stepDis,
			sys.stdout.flush()
		#end if
		jAssign = z[i]
		m, w = refIdxs[i]
		
		q[w,jAssign] -= 1
		n[m,jAssign] -= 1
		k[jAssign] -= 1
		N[m] -= 1
		
		sumR = 0.0
		cummulative = []
		
		for j in range(K):
			sumR += ((q[w,j] + beta) * (n[m,j] + alpha)) / \
					((k[j] + (V*beta)) * (N[m] + (K*alpha)))
			cummulative.append(sumR)
		#end for
		rSample = random.uniform(0, sumR)
		newjAssign = bisect.bisect(cummulative, (rSample,))
		z[i] = newjAssign
		q[w,newjAssign] += 1
		n[m,newjAssign] += 1
		k[newjAssign] += 1
		N[m] += 1
	#end for
	print "done"
	sys.stdout.flush()
	computeNewThi()
	delta = getThiChange()
	oldThi = np.copy(newThi)
	print "delta =", delta
	numEpochs += 1
	print "numEpochs =", numEpochs
#end while

reader = csv.reader(open(INPUT_WORD_LIST, "rb"), delimiter = ",")
WL = [name for line in reader for name in line]
for j in range(K):
	print "\nMost likely words for label %d:"%(j),
	this = newThi[:,j]
	sortedThis = np.argsort(this)[::-1]
	for i in range(min(10, len(sortedThis))):
		print "%s,"%(WL[sortedThis[i]]),
	#end for
	print ""
#end for

computeNewTheta()

print "Saving thi's and theta's ...",
sys.stdout.flush()
np.savetxt(OUTPUT_THIS, newThi, fmt='%.8f', delimiter = ',')
np.savetxt(OUTPUT_THETAS, newTheta, fmt='%.8f', delimiter = ',')
print "done"
sys.stdout.flush()
