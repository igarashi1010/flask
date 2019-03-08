import os
import sys
import models
import nlptools
import gensim
import MeCab
import mysql.connector
import ast
import datetime
import pickle
import pandas as pd
from toolsconfig import dict_path, sentiment_path, model_path, sw_path, idf_path
 
 # # 以下、DB接続関連の関数
 
def connect_db():
    """ データベス接続に接続します """
    con = mysql.connector.connect(
    host = 'localhost',
    user = 'root',
    database = "causal_network",
    password = "opeshitsu"
    )
    return con
 

def main(input_fiile):
    con = connect_db()
    df = pd.read_csv(input_fiile)
    for i, row in df.iterrows():
        layer_num = row["LayerNum"]
        sector = row["Sector"]
        risk = row["RiskSentence"]
        from_date = row["FromDate"]
        to_date = row["ToDate"]
        thre = row["Threshold"]
        direction = row["Direction"]
        use_sentiment = row["UseSentiment"]

        now = datetime.datetime.now()
        date = now.strftime('%Y%m%d')

        data_dict = {
        "Date": date, 
        "RiskSentence": risk,
        "LayerNum": layer_num,
        "Sector": sector,
        "FromDate": from_date,
        "ToDate": to_date,
        "Direction": direction,
        "Thre": thre,
        "UseSentiment": use_sentiment,
        "SentimentDict": sentiment_dict,
        "TestFlag": 0,
        "IdfDict": idf_dict,
        "Stopwords": []
        }

        node_list, edge_list, valid_search_dict, risk_sentiment, node_sentiment_dict, node_sentiment_pair_dict = models.get_relations(con, mecab, model,data_dict, event_tablename, kessan_tablename, topn)

        search_count = len(valid_search_dict.keys())
        search_result = [search_count, node_list, edge_list]
        search_result_str = "探索対象レコード数: {}".format(search_count)
        log_table = "logs"
        ID = models.insert_log(con, log_table, date, layer_num, sector, risk, from_date, to_date, search_result_str, thre, direction, risk_sentiment, use_sentiment)
        result_table = "result"
        models.insert_result(con, result_table, ID, edge_list, date)
        models.draw_graphviz(node_list, edge_list, risk, risk_sentiment, direction, ID, risk, node_sentiment_pair_dict)

if __name__ == '__main__':
    os.environ["PATH"] += os.pathsep + "/home/m2018higarashi/.linuxbrew/bin"
    print("loading model.......")
    model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(model_path)
    # model = gensim.models.word2vec.Word2Vec.load(model_path)
    print("create mecab instance.......")
    mecab = MeCab.Tagger("-d " + dict_path)
    print("loading sentiment dict....")
    sentiment_dict=nlptools.get_sentiment_dict(sentiment_path=sentiment_path)
    event_tablename = "fr_event"
    kessan_tablename = "fr_kessan"
    with open(idf_path, "rb") as f:
        idf_dict = pickle.load(f)

    input_fiile = sys.argv[1]
    main(input_fiile)
    
