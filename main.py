import os
from utils.data_preprocessing import preprocess_all_vtt_files
from utils.config import ROOT_DIR
from model.topic_modeling import *
from utils.data_scrapping import download_playlistCC

#1 Data Scrapping: Download playlists
# playlists = [
#     'https://youtube.com/playlist?list=PLEI1XV90ckT4BK9blWZUItoy8RbXzH8oC&si=IscbqVJjX54QyniU',
#     'https://www.youtube.com/watch?v=QY50cA6Rlnw&list=PLqrLgLCVrIwZgdDbLz1Zt0yboY6diOtH9'
#     'https://www.youtube.com/watch?v=Z18EJxv-Ne8&list=PLCuVYHE7O2A0XGkdhaU_M4-IkEIq5hSOQ'
#     'https://www.youtube.com/playlist?list=PL5ReKt064oo_WziqACFo1YV6lDl6J_JTC',
#     'https://www.youtube.com/playlist?list=PLsvOo4dTDY-GxiGNtS-LCvtEinr7SCDev',
#     'https://www.youtube.com/playlist?list=PLi1026PfRpeXjDH305KzfUDNBSqKA7_Y1',
#     'https://www.youtube.com/playlist?list=PLuHVBmVfl-DCarzH9t3K-NuGBlKs88jUs',
#     'https://www.youtube.com/playlist?list=PL9Mwa4xkeUDRjua13ZcQ8Ru9quTI5HwLV',
#     'https://www.youtube.com/playlist?list=PLNytm2ujrk2kamrb-pjSb-14ThRw0CtaJ',
# ]
# for playlist in playlists:
    # download_playlistCC(playlist)



#2 Data Preprocessing: VVT to TXT, and remove timing
# dataset_dir = os.path.join(ROOT_DIR, 'data', 'input')
# for path in os.listdir(dataset_dir):
#     full_path = os.path.join(dataset_dir, path)
#     preprocess_all_vtt_files(root_dir=full_path, output_dir=os.path.join(ROOT_DIR, 'data', 'working', path))


#3 Topic Modelling
spark = create_spark_session()

# Load dataset as spark Dataframe
preprocced_veri_set = os.path.join(ROOT_DIR, 'data', 'working', '*', '*.txt')  
raw_df = load_dataset(spark, preprocced_veri_set)

# Preprocess Or cleaning the dataset
clean_df = preprocess_data(raw_df)

# Tokenization: raw text converted into words
wordsData = tokenize_data(clean_df)

# Stopwords removal: remove character that are considered noise
wordsData = removeStopWords(wordsData)


wordsData = apply_turkish_stemming(wordsData)

# Vector
featurizedData, countVectorizerModel = vectorize_data(wordsData, vocabSize=10000)
rescaledData = create_tfidf_features(featurizedData)

lda = train_LDA_model(rescaledData, k=10, maxItr=5) 
print('Training done...\n')
analyze_and_interpret_topics(lda, num_topics=10, cvModel=countVectorizerModel)
