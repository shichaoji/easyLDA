from .base import PipelineLDA, main

try:
    import nltk
    nltk.download("stopwords")
except:
    'stopword data downloaded already'


def main():
    print 'executing...'
    main()

