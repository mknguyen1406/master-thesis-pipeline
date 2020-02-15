import pandas as pd

import pickle
import datetime

from gensim import models  # includes multi-core LDA


class LdaModel:

    def __init__(self, dictionary, doc_term_matrix, seed, num_topics):

        self.dictionary = dictionary
        self.doc_term_matrix = doc_term_matrix
        self.seed = seed
        self.num_topics = num_topics
        self.model = None

    def train_model(self):
        """
        create lda models with different random state values
        """

        print("Start time: {}".format(datetime.datetime.now()))

        lda_class = models.LdaMulticore
        lda = lda_class(self.doc_term_matrix, num_topics=self.num_topics, id2word=self.dictionary, passes=20,
                        chunksize=2000, random_state=self.seed)

        print("Finish time: {}".format(datetime.datetime.now()))

        self.model = lda

    # print all topics
    def get_top_n_terms(self, n):
        out = []
        for topic in self.model.print_topics():
            topic_id = topic[0]
            terms_raw = topic[1].split("+")
            terms = [x.split("*")[1] for x in terms_raw]  # get rid of probabilities
            terms = [x.replace("\"", "").replace(" ", "") for x in
                     terms[:n]]  # get rid of quotation marks and whitespaces
            out.append(terms)

        return pd.DataFrame(out)