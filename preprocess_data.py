import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import fasttext
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer
import string
import re
from sklearn import preprocessing
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,auc
from sklearn.model_selection import train_test_split
le = preprocessing.LabelEncoder()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text_data(x):
    lemmatizer = WordNetLemmatizer()
    translator = str.maketrans('', '', string.punctuation)
    text = x.lower()
    text = remove_stopwords(x)
    text = re.sub(r'\d+', '', text)
    text = " ".join(text.split())
    tokenized_text = simple_preprocess(text, deacc=True)
    lemmatized_text = [lemmatizer.lemmatize(a) for a in tokenized_text]
    return ' '.join(lemmatized_text)

def add_label(x):
    return '__label__'+x

def preparetrainingdata(df):
    try:
        X = df.preprocessed_text
        y = df.label
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, shuffle=True)
        df_train_main = pd.DataFrame({'text':X_train,'labels':y_train})
        df_test_main = pd.DataFrame({'text':X_test,'labels':y_test})

        x2 = df_test_main.text
        y2 = df_test_main.labels
        X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, y2, stratify=y2, test_size=0.1)
        df_test_1 = pd.DataFrame({'text':X_train2,'labels':y_train2})
        df_test_2 = pd.DataFrame({'text':X_test2,'labels':y_test2})
        df_test_2.to_csv('validation_sample_1.csv', index=True)

        df_test_2_labels = df_test_2.labels.value_counts().rename_axis('unique_values').reset_index(name='counts')
        df_test_2_labels['document'] = df_test_2_labels['unique_values'].apply(lambda x: le.inverse_transform([int(re.findall('\d+',x)[0])])[0])
        df_test_2_labels.to_csv('doc_lookupvalues.csv', index = False)

        df_train_main['data'] = df_train_main['labels'] + " " + df_train_main['text']
        df_test_1['data'] = df_test_1['labels'] + " " + df_test_1['text']
        df_train_main.to_csv('fasttext_train.txt', columns=['data'], header=False,index=False)
        df_test_1.to_csv('fasttext_test.txt', columns=['data'], header=False,index=False)
    except Exception as e:
        print('Exception in preparetrainingdata---->',e)

def data_preparation(data_path):

    try:
        df = pd.read_csv(data_path)
        df.reset_index(inplace = True)
        df = df[['Raw_Text','Doctype']]
        df['preprocessed_text'] = df['Raw_Text'].apply(preprocess_text_data)
        df = df[['preprocessed_text','Doctype']]
        val_counts = df['Doctype'].value_counts()
        val_counts.to_csv('documents_sample_counts.csv',index=False, header=True)
        df = df[['preprocessed_text','Doctype']]
        df['label'] = le.fit_transform(df['Doctype'])
        df = df.drop(columns='Doctype')
        df['label'] = df['label'].apply(lambda x:add_label(str(x)))
        preparetrainingdata(df)
    except Exception as e:
        print('Exception in data_preparation===>',e)
        return 'Error has been occurred'