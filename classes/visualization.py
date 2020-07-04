import pandas as pd
from collections import Counter
from wordcloud import WordCloud as WordCloudClass
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import itertools


class Viz:

    def __init__(self, wc_file_path, word_no, topic_labels, rg_file_path=None, factor_list=None):

        self.seed = 0

        # For word clouds
        self.wc_data_file_path = wc_file_path
        self.word_no = word_no
        self.topic_labels = topic_labels
        self.topic_number = len(topic_labels)
        self.word_clouds = None

        # For regression plots
        self.rg_file_path = rg_file_path
        self.factor_list = factor_list

    def generate_word_clouds(self):

        # Transform loads into artificial frequencies and reset index
        def hlp_generate_frequency(load):
            return round(load * 1000)

        # Generate world cloud for one topic
        def hlp_generate_word_cloud(topic_no):

            # Get data for specific topic
            df_topic = df[df["topic"] == topic_no]

            # Specify columns to be kept
            columns = ["word", "load"]
            df_topics_overall = df_topic[columns]

            # Group values by words over all time periods
            df_grouped = df_topics_overall.groupby(["word"]).sum()

            # Sort descending and grab top n words
            df_sorted = df_grouped.sort_values("load", ascending=False)
            df_top_n = df_sorted[:self.word_no]

            freq_values = df_top_n["load"].apply(lambda load: hlp_generate_frequency(load))
            df_top_n["freq"] = freq_values

            # Convert word column from index to normal column
            df_top_n = df_top_n.reset_index()

            # Generate list with words considering the artificial frequency
            freq_list = df_top_n["freq"]
            word_list = df_top_n["word"]
            topic_word_list = []

            for i in range(len(word_list)):
                word = word_list[i]
                freq = freq_list[i]

                word_freq = [word] * int(freq)
                topic_word_list = topic_word_list + word_freq

            # Initialize and update counter
            result = Counter()
            result.update(topic_word_list)

            # Create custom black color map
            black_cm = ListedColormap([0, 0, 0, 1])  # black

            # Generate word cloud
            wc = WordCloudClass(background_color="white", colormap=black_cm, max_font_size=80, random_state=self.seed,
                                max_words=self.word_no, width=400, height=300)

            return wc.fit_words(result)

        # Read file
        df = pd.read_csv(self.wc_data_file_path, sep=";", decimal=",")

        self.word_clouds = []

        for i in range(self.topic_number):
            # Generate world cloud
            word_cloud = hlp_generate_word_cloud(i)

            self.word_clouds.append(word_cloud)

    def generate_word_cloud_viz(self, file_path):

        if self.word_clouds:

            fig, axs = plt.subplots(nrows=4, ncols=5, figsize=[15, 10])
            fig.tight_layout()

            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.2)

            for row, col in itertools.product(range(4), range(5)):
                # Get word cloud
                wc_id = 5 * row + col
                word_cloud = self.word_clouds[wc_id]

                axs[row][col].imshow(word_cloud, interpolation="bilinear")
                axs[row][col].set_yticklabels([])
                axs[row][col].set_xticklabels([])
                axs[row][col].set_title(f"ID: {wc_id}; {self.topic_labels[wc_id]}")

            fig.savefig(file_path)

            print(f"Figure saved to {file_path}\n")

        else:
            print("Generate word clouds first")

    def create_regression_plot(self):

        print("Hello")


