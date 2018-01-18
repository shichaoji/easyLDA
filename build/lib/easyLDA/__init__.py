try:
    import nltk
    nltk.download("stopwords")
    nltk.download("wordnet")
except:
    'stopword data downloaded already'
    
from .base import PipelineLDA
from .base import main as LDA_main



def main():
    print 'executing...'
    LDA_main()

