import pandas as pd
from tqdm import tqdm  # to get progress bar with apply function

import spacy  # for data cleaning
from translate import Translator  # for translation
import re

from gensim.models import Phrases
from gensim import corpora  # to create a dictionary out of all words


class DataCleaner:

    def __init__(self, input_path, output_path, file_name, sheet_name, date_col_index, id_col, text_col, date_col,
                 keep_pos, translate=False, add_bigrams=True, bigram_min_count=10, dict_no_below=5, dict_no_above=0.5,
                 additional_stop_words=None):

        self.input_path = input_path
        self.output_path = output_path
        self.file_name = file_name
        self.sheet_name = sheet_name
        self.date_col_index = date_col_index
        self.id_col = id_col
        self.text_col = text_col
        self.date_col = date_col
        self.keep_pos = keep_pos
        self.translate = translate
        self.add_bigrams = add_bigrams
        self.bigram_min_count = bigram_min_count
        self.dict_no_below = dict_no_below
        self.dict_no_above = dict_no_above
        self.additional_stop_words = additional_stop_words

    def read_data(self):

        # Read raw data
        file_path = self.input_path + self.file_name + ".xlsx"
        df = pd.read_excel(file_path, sheet_name=self.sheet_name,
                           parse_dates=self.date_col_index, infer_datetime_format=True)

        # Sort by date
        df = df.sort_values(self.date_col).reset_index(drop=True)

        print(f"Raw data read from {file_path}")

        return df

    def clean_data(self, df):

        nlp = spacy.load('en_core_web_sm')

        # Add domain-specific stop words
        for stop_word in self.additional_stop_words:
            nlp.Defaults.stop_words.add(stop_word)

        # Get stop_word_list
        stop_word_list = list(nlp.Defaults.stop_words)

        # Get rid of columns without id, objective, or date
        target_col = [self.id_col, self.date_col, self.text_col]

        print(f"No. of row before dropping NAs: {df.shape[0]}")

        df = df.dropna(subset=target_col).reset_index(drop=True)
        df = df[target_col]

        print(f"No. of row before dropping NAs: {df.shape[0]}")

        text_data = df[self.text_col]

        # clean up your text and generate list of words for each document
        def clean_up(text):
            text_out = []

            # data cleaning with spacy; returns array of tokenized document
            try:
                # Remove leading and tailing whitespaces
                text = text.strip()

                # Lowercase
                text = text.lower()

                # Replace multiple whitespaces with single one
                text = re.sub(' +', ' ', text)

                # Parse with spacy
                doc = nlp(text)

                # only if translation is required - translate to english
                # if self.translate:
                #     from_lang = doc._.language["language"]
                #     to_lang = 'en'
                #
                #     if from_lang != 'en':
                #         translator = Translator(to_lang=to_lang, from_lang=from_lang)
                #         translation = translator.translate(text)
                #         doc = self.nlp(translation)

                for token in doc:
                    # only keep words with following criteria:
                    # not a stop word, alphabetic characters, at least length of 3, and not in POS removal list

                    # Get word lemma
                    lemma = token.lemma_
                    pos = token.pos_
                    is_alpha = token.is_alpha

                    # Only keep if not in POS removal list, not stop word, is alphabetic, length at least 3
                    if pos in self.keep_pos and len(lemma) > 2 and is_alpha and lemma not in stop_word_list:

                        # Append to array
                        text_out.append(lemma)

            except:
                print(f"Error with text '{text}'")

            return text_out

        # Create and register a new `tqdm` instance with `pandas`
        tqdm.pandas()

        # Apply cleaning to all documents
        data_clean = text_data.progress_apply(lambda text: clean_up(text))

        print("Finished cleaning")

        return df, data_clean

    def add_bigrams_to_data(self, data_clean):

        if self.add_bigrams:
            # Add bigrams and trigrams to docs (only ones that appear 10 times or more).
            bigram = Phrases(data_clean, min_count=self.bigram_min_count)

            for idx in range(len(data_clean)):
                for tok in bigram[data_clean[idx]]:
                    if '_' in tok:
                        # Token is a bigram, add to document.
                        data_clean[idx].append(tok)

        print("Bigrams added")

        return data_clean

    def convert_data_clean(self, df, data_col):

        # return list from string
        def to_list(text):
            return text.replace("[", "").replace("]", "").replace("'", "").replace(" ", "").split(",")

        # Apply cleaning function
        data_clean = df[data_col].apply(lambda text: to_list(text))

        return data_clean

    def create_dictionary(self, data_clean):

        dictionary = corpora.Dictionary(data_clean)

        # Filter out words that occur less than 10 documents, or more than 20% of the documents.
        print('Number of unique words before removing rare and common words:', len(dictionary))
        dictionary.filter_extremes(no_below=self.dict_no_below, no_above=self.dict_no_above)
        print('Number of unique words after removing rare and common words:', len(dictionary))

        return dictionary

    def save_dictionary(self, dictionary, file_path):

        # Save dictionary
        dictionary.save(file_path)

        print(f"Dictionary saved to {file_path}")

    def load_dictionary(self, file_path):

        # Save dictionary
        dictionary = corpora.Dictionary.load(file_path)

        print(f"Dictionary loaded from {file_path}")

        return dictionary

    def create_merged_dictionary(self, files):

        # return list from string
        def to_list(text):
            return text.replace("[", "").replace("]", "").replace("'", "").replace(" ", "").split(",")

        dict_list = []
        # Get already cleaned data from files
        for file in files:
            print(f"Start processing {file}")

            df_file = pd.read_csv(self.output_path + file + "_clean.csv", index_col=0)

            # Apply cleaning function
            data_clean = df_file["data_clean"].apply(lambda text: to_list(text))

            # Create dictionary
            dictionary = corpora.Dictionary(data_clean)

            # Filter out words that occur less than 10 documents, or more than 20% of the documents.
            print('Number of unique words before removing rare and common words:', len(dictionary))

            dictionary.filter_extremes(no_below=10, no_above=0.2)
            print('Number of unique words after removing rare and common words:', len(dictionary))

            dict_list.append(dictionary)

        print("All dictionaries created")

        # Merge dictionaries from list
        dictionary = dict_list[0]
        for i in range(len(dict_list) - 1):
            dictionary.merge_with(dict_list[i + 1])

        print("All dictionaries merged")

        return dictionary

    def create_doc_term_matrix(self, dictionary, data_clean):

        doc_term_matrix = [dictionary.doc2bow(doc) for doc in data_clean]

        print("Doc-term-matrix created")

        return doc_term_matrix

    def save_df_clean(self, df_clean):

        # Save to output folder
        output_file_path = self.output_path + self.file_name + "_clean.csv"
        df_clean.to_csv(output_file_path)

        print(f"Clean data frame saved to {output_file_path}")
