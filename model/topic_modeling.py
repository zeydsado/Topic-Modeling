from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, lower, col, udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer
from pyspark.ml.clustering import LDA, LDAModel
from snowballstemmer import TurkishStemmer

def create_spark_session():
    title = "TV Drama Series Topic Modeling"
    spark = SparkSession.builder.appName(title).getOrCreate()
    return spark

def load_dataset(spark, data_path):
    rdd = spark.sparkContext.wholeTextFiles(data_path)
    df = rdd.toDF(["file_path", "text"])
    return df

def preprocess_data(df):
    df = df.withColumn('clean_text', regexp_replace(col('text'), "[^a-zA-ZçÇğĞıİöÖşŞüÜ\\s]", ""))
    df = df.withColumn('clean_text', lower(col('clean_text')))
    return df


def tokenize_data(df):
    tokenizer = Tokenizer(inputCol='clean_text', outputCol='words')
    wordsData = tokenizer.transform(df)
    return wordsData

def removeStopWords(wordsData):
    stopWords = stopWords=StopWordsRemover.loadDefaultStopWords("turkish")
    remover = StopWordsRemover(inputCol='words', outputCol='filtered', stopWords=stopWords)
    wordsData = remover.transform(wordsData)  
    return wordsData


def vectorize_data(wordsData, vocabSize=10000):
    countVectorizer = CountVectorizer(inputCol='filtered', outputCol='rawFeatures', vocabSize=vocabSize)
    countVectorizerModel = countVectorizer.fit(wordsData)
    featurizedData = countVectorizerModel.transform(wordsData)
    return featurizedData, countVectorizerModel


def apply_turkish_stemming(wordsData):
    stemmer = TurkishStemmer()
    stemmed_words = udf(lambda words: [stemmer.stem(word) for word in words], ArrayType(StringType()))
    wordsData = wordsData.withColumn('words', stemmed_words(col('words')))
    return wordsData

def create_tfidf_features(featurizedData):
    idf = IDF(inputCol='rawFeatures', outputCol='features') 
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    return rescaledData

def train_LDA_model(rescaledData, k=10, maxItr=10):
    ldaModel = LDA(k=k, maxIter=maxItr).fit(rescaledData)
    return ldaModel

def analyze_and_interpret_topics(ldaModel, num_topics, cvModel=None):
    topics_data = ldaModel.describeTopics(num_topics)
    vocab = cvModel.vocabulary if cvModel is not None else None

    for topic_idx, topic in enumerate(topics_data.collect()):
        print(f"Topic {topic_idx}:")
        words = [vocab[word_idx] for word_idx in topic['termIndices']] if vocab is not None else topic['termIndices']
        print(f"Top words: {words}")

        interpretation = interpret_topic(words)
        print(f"Interpretation: {interpretation}\n")

def interpret_topic(words):
    # Interpret topics based on common themes in Turkish TV dramas
    if 'aşk' in words or 'sevgi' in words:
        return "Aşk ve Romantizm"  # Love and Romance
    elif 'aile' in words or 'ev' in words:
        return "Aile ve Ev"  # Family and Home
    elif 'savaş' in words or 'mücadele' in words:
        return "Savaş ve Mücadele"  # War and Struggle
    elif 'cinayet' in words or 'suç' in words:
        return "Suç ve Gizem"  # Crime and Mystery
    elif 'kraliyet' in words or 'padişah' in words:
        return "Kraliyet ve Siyaset"  # Royalty and Politics
    elif 'arkadaş' in words or 'dostluk' in words:
        return "Arkadaşlık ve Dostluk"  # Friendship
    elif 'para' in words or 'iş' in words:
        return "Para ve İş"  # Money and Work
    elif 'hastalık' in words or 'sağlık' in words:
        return "Hastalık ve Sağlık"  # Illness and Health
    elif 'intikam' or words or 'nefret' in words:
        return "İntikam ve Nefret"  # Revenge and Hatred
    else:
        return "Genel Drama"  # General Drama
