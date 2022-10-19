__author__ = '{Esra Dönmez}'

import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

class PrepareData:
    """
    A class to preprocess the data.
    """

    def __init__(self, columns=None):
        # initialize stopwords, lemmatizer and stemmer
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        self.stemmer = nltk.stem.porter.PorterStemmer()
        # set the columns if the input is pandas dataframe
        if columns is not None:
            self.columns = columns[1:]

    def preprocess_text(self, text, flg_clean=False, flg_stemm=False, flg_lemm=False, stopwords=None):
        """
        Preprocesses a text with given conditions.

        Args:
            text: text to be preprocessed
            flg_clean: whether to remove non-word characters
            flg_stemm: whether to stemm
            flg_lemm: whether to lemmatize
            stopwords: stopwords to be removed
        
        Returns:
            list of preprocessed tokens
        """
        # remove anything that is not word or space,
        if flg_clean:
            text = re.sub(r'[^\w\s]', '', str(text))

        # strip and lowercase
        # whitespace-tokenize - split on space
        tokenized = str(text).strip().lower().split()

        # remove Stopwords
        if stopwords is not None:
            tokenized = [word for word in tokenized if word not in
                         stopwords]

        # Stemming (remove -ing, -ly, ...)
        if flg_stemm == True:
            ps = nltk.stem.porter.PorterStemmer()
            tokenized = [ps.stem(word) for word in tokenized]

        # Lemmatisation (convert the word into root form)
        if flg_lemm == True:
            lem = nltk.stem.wordnet.WordNetLemmatizer()
            tokenized = [lem.lemmatize(word) for word in tokenized]

        # convert list back to string
        # text = " ".join(tokenized)
        return tokenized

    def preprocess_data(self, lst_data, flg_clean, flg_stemm, flg_lemm):
        """
        Preprocesses a list containing text.

        Args:
            lst_data: list to be preprocessed
            flg_clean: whether to remove non-word characters
            flg_stemm: whether to stemm
            flg_lemm: whether to lemmatize
        
        Returns:
            list containing lists of preprocessed tokens
        """
        data = []
        for line in lst_data:
            example = self.preprocess_text(line, flg_clean, flg_stemm,
                                           flg_lemm, self.stopwords)
            data.append(example)

        return data
