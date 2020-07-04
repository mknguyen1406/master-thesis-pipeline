from classes.data_cleaning import DataCleaner
from classes.data_aggregating import DataAggregator
from classes.lda_modeling import LdaModel
from classes.dtm_modeling import DtmModel
from classes.dtm_evaluation import Evaluator
from classes.data_exploration import Explorer
import pandas as pd
from classes.visualization import Viz

from gensim import corpora

import datetime

if __name__ == '__main__':
    wc_file_path = "output/topics/all_topics.csv"
    word_no = 50
    topic_labels = [
        'Agricultural Science',
        'Energy Science',
        'Information Systems',
        'Chemistry',
        'Aerodynamics',
        'Cell Research',
        'Research Programmes',
        'Material Science',
        'Astrophysics',
        'European Development',
        'Social Science',
        'Health Science',
        'Genetic Research',
        'Electronics and Photonics',
        'Neuroscience',
        'Quantum Physics',
        'Molecular Biology',
        'Energy Innovations',
        'Software Engineering',
        'Climate Science',
    ]

    viz = Viz(wc_file_path, word_no, topic_labels)

    viz.generate_word_clouds()
    viz.generate_word_cloud_viz("images/4_word_clouds.png")