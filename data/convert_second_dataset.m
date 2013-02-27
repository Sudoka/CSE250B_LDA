% Purpose: Load second dataset into same format as classic400.mat
%
% David Larson
% CSE 250B, Winter 2013, UCSD


%% Load data
data = importdata('docword_kos_400docs.txt', ' ', 3);

% index 1: docID (int)
% index 2: wordID (int)
% index 3: count (int)
docwords = data.data;
clear data

%% Organize data into 2D array where rows=docID, col=wordID

n_docs = 400;
n_vocab = 6906;

wordlist = zeros(n_docs, n_vocab);

for row=1:length(docwords(:,1))
    i = docwords(row, 1)
    j = docwords(row, 2)
    count = docwords(row, 3);
    
    wordlist(i, j) = count;
end

%% Load vocab
vocab_data = importdata('vocab_kos.txt');