import sys
import os
import re
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import time
# from PySide2.QtCore import Qt
# from PySide2.QtWidgets import (QApplication, QMainWindow, QGridLayout, QPushButton, QLabel, QWidget,
#                                QBoxLayout, QComboBox, QLineEdit, QSpinBox, QFrame,
#                                QAbstractSpinBox, QFileDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGridLayout, QPushButton, QLabel, QWidget,
                               QBoxLayout, QComboBox, QLineEdit, QSpinBox, QFrame,
                               QAbstractSpinBox, QFileDialog)

import webbrowser

os.environ['QT_PLUGIN_PATH']='C:/Users/lpatterson/AppData/Local/Continuum/anaconda3/Library/plugins/PyQt5'

def run_lda(settings):
    file_name = settings.file_name
    if file_name[-5:] == '.xlsx' or file_name[-4:] == '.xls':
        df = pd.read_excel(file_name)
    else:
        df = pd.read_csv(file_name,low_memory=False)
    df = df.astype({'id': int}).sample(frac=1)

    print('Cleaning text')
    print(time.process_time())
    X = df['text']
    stemmer = SnowballStemmer("english")
    X = [' '.join([stemmer.stem(word) for word in x.split()]) for x in X]
    X = [re.sub(r'[^a-zA-Z\s]', '', x) for x in X]

    print('Vectorizing text')
    print(time.process_time())
    count_vectorizer = CountVectorizer(max_df=0.8, min_df=0.01, stop_words='english')
    X = count_vectorizer.fit_transform(X)

    lda = LatentDirichletAllocation(n_components=settings.n_components, learning_method=settings.learning_method)
    print('Training model')
    print(time.process_time())
    lda.fit(X)

    print('Getting topic distributions')
    print(time.process_time())
    topic_distributions = lda.transform(X)
    components, topic_distributions = sort_topics(topic_distributions, lda.components_)
    df['topic_distributions'] = topic_distributions.tolist()
    closest_topics = get_closest_topics(topic_distributions)
    df['closest_topics'] = closest_topics

    topic_data = get_topic_documents(df, components, count_vectorizer, topic_distributions)

    return df.set_index('id'), topic_data


def sort_topics(topic_distributions, components):
    order = np.flip(np.argsort(np.sum(topic_distributions, axis=0)), axis=0)
    return components[order], topic_distributions[:, order]


def get_closest_topics(topic_distributions):
    s = topic_distributions.shape
    result = [[] for _ in range(s[0])]
    indices = np.argwhere(topic_distributions > 1/s[1])

    for i in indices:
        result[i[0]].append(i[1])

    return result


def get_topic_documents(df, components, count_vectorizer, topic_distributions):
    topics = []
    for topic in components:
        top_words_indices = np.flip(np.argsort(topic), axis=0)[:20]
        top_words = [count_vectorizer.get_feature_names()[i] for i in top_words_indices]
        top_words_values = topic[top_words_indices].tolist()
        topic_name = ', '.join(top_words[:3])
        topics.append({'name': topic_name, 'top_words': top_words, 'top_words_values': top_words_values, 'documents': []})

    for i in range(df.shape[0]):
        for j in df.iloc[i]['closest_topics']:
            topics[j]['documents'].append(int(df.iloc[i]['id']))

    for i, topic in enumerate(topics):
        topic['number_of_docs'] = len(topic['documents'])
        topic['average_likelihood'] = np.sum(topic_distributions[:, i]) / topic_distributions.shape[0]
        print(topic['average_likelihood'])

        top_documents = np.flip(np.argsort(topic_distributions[:, i]), axis=0)[:20]
        topic['top_documents'] = list(df.iloc[top_documents]['id'])

    return topics


class GUI(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QGridLayout()
        self.main_layout.setHorizontalSpacing(0)
        self.main_layout.setVerticalSpacing(0)
        #self.main_layout.setMargin(10)
        self.central_widget.setLayout(self.main_layout)

        self.settings = Settings()

        self.file_dialog = FileInput('Corpus File:')
        self.main_layout.addWidget(self.file_dialog, 0, 0)

        self.tm_method = QComboBox()
        self.tm_method.addItem('LDA')
        self.tm_method.setCurrentIndex(0)
        self.tm_method_area = InputArea('Topic Modeling Method:', self.tm_method)
        self.main_layout.addWidget(self.tm_method_area, 1, 0)

        self.n_components = QSpinBox()
        self.n_components.setRange(0, 1000)
        self.n_components.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.n_components.setValue(self.settings.n_components)
        self.n_components_area = InputArea('Number of Components:', self.n_components)
        self.main_layout.addWidget(self.n_components_area, 2, 0)

        self.learning_method = QComboBox()
        self.learning_method.addItems(['Online', 'Batch'])
        self.learning_method.setCurrentIndex(0)
        self.learning_method_area = InputArea('Learning Method:', self.learning_method)
        self.main_layout.addWidget(self.learning_method_area, 3, 0)

        self.run_analysis = QPushButton('Run Analysis')
        self.run_analysis.setObjectName('run_analysis')
        self.run_analysis.clicked.connect(self.submit_parameters)
        self.main_layout.addWidget(self.run_analysis, 4, 0)

        self.setWindowTitle("Topic Modeling GUI")

    def submit_parameters(self):
        print('Updating settings')
        self.settings.file_name = self.file_dialog.field.text()
        self.settings.tm_method = self.tm_method.currentText()
        self.settings.n_components = self.n_components.value()
        self.settings.learning_method = self.learning_method.currentText().lower()
        documents, topics = run_lda(self.settings)
        print(documents)
        script = '''
                    topics = {};
                    documents = {};
                    '''.format(str(topics), str(documents.to_dict(orient='index')))

        with open("templates/tm_results_template.html") as fp:
            soup = BeautifulSoup(fp)

        soup.find_all('script')[-1].append(script)

        result_file = os.path.join(os.getcwd(), 'results.html')
        print('Saving to {}'.format(result_file))

        with open(result_file, 'w') as fp:
            fp.write(str(soup))

        webbrowser.open(result_file, new=2)


class InputArea(QFrame):
    def __init__(self, name, field):
        QFrame.__init__(self)
        self.setObjectName('input_area')
        self.layout = QBoxLayout(QBoxLayout.LeftToRight)
        self.layout.setSpacing(0)
        #self.layout.setMargin(5)
        self.label = QLabel(name)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label.setObjectName('input_label')
        self.field = field
        self.field.setObjectName('input_field')
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.field)
        self.setLayout(self.layout)


class FileInput(InputArea):
    def __init__(self, name, button_name='Browse'):
        file_input = QLineEdit()
        file_input.setMaximumWidth(200)
        InputArea.__init__(self, name, file_input)
        self.browse_button = QPushButton(button_name)
        self.browse_button.setObjectName('browse_button')
        self.browse_button.clicked.connect(self.get_file_name)
        self.browse_button.setMaximumWidth(80)
        self.layout.addWidget(self.browse_button)
        self.file_dialog = QFileDialog()

    def get_file_name(self):
        file_name = self.file_dialog.getOpenFileName(self, caption='Select File', filter='Spreadsheet (*.csv *.xlsx *.xls)')
        if file_name[0] != '':
            self.field.setText(file_name[0])


class Settings:
    def __init__(self):
        self.file_name = ''
        self.tm_method = 'LDA'
        self.n_components = 15
        self.learning_method = 'online'


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet('QMainWindow { background: #333333 }'
                      'QWidget { font-size: 14px }'
                      'QLabel#input_label { color: white; margin-right: 5px; }'
                      '#input_field { background: white; color: black; border: 0px solid black; padding: 1px 2px; }'
                      'QWidget#run_analysis { background: #339933; color: white; font-weight: bold; border: 1px solid #194d19; padding: 5px 0px;}'
                      'QWidget#browse_button { background: white; border: 1px solid white; padding: 1px 5px; margin-left: 5px; } ')
    window = GUI()
    window.setGeometry(100, 100, 400, 200)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
