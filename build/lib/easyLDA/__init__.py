# coding: utf-8

#import sys
#try:
#    reload(sys)
#    sys.setdefaultencoding("utf-8")
#except:
#    print('python3')
    
import nltk
try:
    from nltk.corpus import stopwords
    from nltk.stem import wordnet
    print 'stopword & wordnet data downloaded already'
except:
    print 'downloading nltk data'
    nltk.download("stopwords")
    nltk.download("wordnet")

    
from .base import PipelineLDA
from .base import main as LDA_main



def main():
    print 'executing...'
    LDA_main()

