#Author: Nicholas Low
#Date: 11/2/2015
#Description: Project 3 : Naive Bayes Sentiment Analysis
#Takes in two training files - rotten and fresh
#Trains a naive bayes model
#Prompt user for a testing file name to compute the probability.
#Display the two probabilities on the screen and the verdict
import sys
import nltk
import string
from nltk.probability import FreqDist
from nltk.tokenize import TweetTokenizer
#Global constants for fresh and rotten based on top 10 box office movies.
freshp = 0.688
rottenp = 0.312
mconst = 100.0

def preprocess(text, write):
	fw = open(write, 'w')
	#Takes away most punctuation and converts apostrophes to ANSI
	for line in text.readlines():
		pl = line.lower()
		pl = pl.translate(str.maketrans("","", '•…­“”!"#$%&()*+,–—-./:;<=>?@[\\]^_{|}~'))
		pl = pl.replace('’', '\'')
		pl = pl.replace('‘', '\'')
		pl.rstrip()
		fw.write(pl+'\n')
			
def wordfreq(text, write):
	fw = open(write, 'w')
	fdist = FreqDist()
	freq = {}
	freq['UNK'] = [0,0.0]
	linecnt = 0
	#Get lines count from text file
	for line in text.readlines():
		linecnt += 1
	text.seek(0)
	#Determine held out training set
	heldout = linecnt * .1
	#Train non-held out set
	for l in range(0, linecnt - int(heldout)):
		line = text.readline()
		for word in TweetTokenizer().tokenize(line):
			fdist[word] += 1
	#Train held out set
	for line in text.readlines():
		for word in TweetTokenizer().tokenize(line):
			if fdist[word] != 0:
				fdist[word] += 1
			else:
				fdist['UNK'] += 1
	#Determine conditional probabilities
	for word in fdist.most_common():
		prob = float(fdist[word[0]])/fdist.N()
		freq[word[0]] = [fdist[word[0]], prob]
		fw.write(word[0]+' '+str(fdist[word[0]])+' '+str(prob)+'\n')
	return freq
	
def naive_bayes(test, fresh, rotten):
	#Grabs global constants to avoid rewriting them
	freshprob = freshp
	rottenprob = rottenp
	#Gets preprocessed text
	preprocess(open(test), 'PreprocessedText/pp_testfile.txt')
	tf = open('PreprocessedText/pp_testfile.txt')
	#Reads in each line of the test file and calculates probabilities for
	#fresh and rotten Naive Bayes models.
	for line in tf.readlines():
		for word in TweetTokenizer().tokenize(line):
			try:
				freshprob = fresh[word][1] * freshprob * mconst
			except KeyError:
				freshprob = fresh['UNK'][1] * freshprob * mconst
			try:
				rottenprob = rotten[word][1] * rottenprob * mconst
			except KeyError:
				rottenprob = fresh['UNK'][1] * rottenprob * mconst
	return (freshprob, rottenprob)

if __name__ == "__main__":
	#Ask user for two training file names
	f1 = input('Fresh training file name: ')
	f2 = input('Rotten training file name: ')
	#Preprocess the text from training files
	preprocess(open(f1), 'PreprocessedText/pp_freshreviews.txt')
	preprocess(open(f2), 'PreprocessedText/pp_rottenreviews.txt')
	#Grabs preprocessed text
	fp = open('PreprocessedText/pp_freshreviews.txt')
	fn = open('PreprocessedText/pp_rottenreviews.txt')
	#Determines and outputs word probabilities and frequencies
	freshfreq = wordfreq(fp, 'Frequencies/prob_freshrev.txt')
	rottenfreq = wordfreq(fn, 'Frequencies/prob_rottenrev.txt')
	#Ask user for test file
	ftest = input('Input test file name: ')
	outcome = naive_bayes(ftest, freshfreq, rottenfreq)
	#Whichever probabilities is higher is the correct prediction.
	if outcome[0] > outcome[1]:
		print('Outcome: Fresh Review')
	else:
		print('Outcome: Rotten Review')
	print('Fresh Probability: '+str(outcome[0])+'\nRotten Probability: '+str(outcome[1]))
	