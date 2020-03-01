from classes.data_cleaning import DataCleaner
from classes.data_aggregating import DataAggregator
from classes.lda_modeling import LdaModel
from classes.dtm_modeling import DtmModel
from classes.dtm_evaluation import Evaluator
# from classes.data_exploration import Explorer
import pandas as pd
from classes.word_cloud import WordCloud

from gensim import corpora

import datetime

if __name__ == '__main__':

    file_path = "output/topics/all_topics.csv"
    word_no = 50
    topic_no = 20

    wc = WordCloud(file_path, word_no, topic_no)

    wc.generate_word_clouds()
    wc.generate_visualization("images/3_word_clouds.png")

