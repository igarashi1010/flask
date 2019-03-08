import os
import sys
import models
import nlptools
from flask import Flask, redirect, url_for, render_template, request, g
import gensim
import MeCab
import mysql.connector
import ast
import datetime
import pickle

from toolsconfig import dict_path, sentiment_path, model_path, sw_path, idf_path

app = Flask(__name__)
app.config.from_pyfile('config.cfg')

os.environ["PATH"] += os.pathsep + "/home/m2018higarashi/.linuxbrew/bin"
 
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
 
@app.route('/')
def index():
    # 初回のみテーブル作成
    create_sql = """
CREATE TABLE IF NOT EXISTS {TABLENAME}  (
RiskID int AUTO_INCREMENT NOT NULL PRIMARY KEY, 
RiskSentence text,
Sector VARCHAR(20),
LayerNum VARCHAR(20),
AnalyzeDate date,
FromDate date,
ToDate date,
SearchResult text,
Threshold float, 
Direction VARCHAR(20),
UseSentiment VARCHAR(20),
RiskSentiment float
);
    """.format(TABLENAME=log_table)
    """dbとの接続開始"""
    con = connect_db()
    cur = con.cursor()

    # cur.execute(create_sql)
    """ 一覧画面 """
    select_sql = """
    SELECT * from {TABLENAME}
    """.format(TABLENAME=log_table)

    l = models.query2list(con, select_sql)
    l.reverse()
    return render_template('index.html',l=l)
 
@app.route('/create')
def create():
    con = connect_db()
    l = models.get_sectors(con)
    """ 新規作成画面 """
    return render_template('edit.html', l=l)
  

@app.route('/analysis', methods=['POST'])
def analysis():
    con = connect_db()

    # date = request.form['date']
    now = datetime.datetime.now()
    date = now.strftime('%Y%m%d')
    layer_num = int(request.form['layer_num'])
    sector = request.form['sector']
    risk = request.form['risk']
    from_date = request.form['from_date']
    to_date = request.form['to_date']
    thre = float(request.form['thre'])
    direction = request.form['direction']
    use_sentiment = int(request.form['use_sentiment'])

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
    "TestFlag": test_flag,
    "IdfDict": idf_dict,
    "Stopwords": []
    }
    
    node_list, edge_list, valid_search_dict, risk_sentiment, node_sentiment_dict, node_sentiment_pair_dict = models.get_relations(con, mecab, model, data_dict, 
        event_tablename=event_table, kessan_tablename=kessan_table, topn=5)

    search_count = len(valid_search_dict.keys())
    search_result = [search_count, node_list, edge_list]
    # search_result_str = "test"
    search_result_str = "探索対象レコード数: {}".format(search_count)


    """ 検索結果に代入処理 """
    con = connect_db()
    ID = models.insert_log(con, log_table, date, layer_num, sector, risk, from_date, to_date, search_result_str, thre, direction, risk_sentiment, use_sentiment)
    models.insert_result(con, result_table, ID, edge_list, date)

    """pngファイルにネットワークを作成"""
    models.draw_graphviz(node_list, edge_list, risk, risk_sentiment, direction, ID, risk, node_sentiment_pair_dict)

    return redirect(url_for('view_result', ID=ID))
 
 
@app.route('/delete/<ID>', methods=['GET','POST'])
def delete_result(ID):
    con = connect_db()
    """dbとの接続開始"""
    models.delete(con, log_table, ID)

    """結果のpngファイルも消去"""
    delete_path = "./static/results/{ID}.png".format(ID=ID)
    if os.path.exists(delete_path):
        os.remove(delete_path)

    return redirect(url_for('index'))
    

@app.route('/view/<ID>', methods=['GET'])
def view_result(ID):
    """ 結果参照処理 """
    con = connect_db()
    condition = models.get_data(con, log_table, ID)[0]
    risk = condition["RiskSentence"]
    edges = models.get_data(con, result_table, ID)
    img_path = "/static/results/{ID}_{RISK}.png".format(ID=ID, RISK=risk)
    return render_template('view.html', condition=condition, edges=edges, img_path=img_path)
 
@app.route('/view/<ID>', methods=['POST'])
def annotate(ID):
    con = connect_db()
    edges = models.get_data(con, result_table, ID)
    if edges:
        edge_IDs = [edge["EdgeID"] for edge in edges]
        for edge_ID in edge_IDs:
            res = request.form["{}".format(edge_ID)]
            models.insert_annotate(con, annotation_table, edge_ID, res)
    return redirect(url_for('view_result',ID=ID))

if __name__ == '__main__':
    print("create mecab instance...")
    value = sys.argv
    test_flag=0
    if len(value) > 1:
        print("TEST MODE")
        test_flag = 1
        mecab = "test"
        model = "test"

    else:
        print("loading model.......")
        mecab = MeCab.Tagger("-d " + dict_path)
        model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(model_path)
        # model = gensim.models.word2vec.Word2Vec.load(model_path)
    # model = gensim.models.word2vec.KeyedVectors.load_word2vec_format(model_path)
    print("loading sentiment dict....")
    sentiment_dict = nlptools.get_sentiment_dict(sentiment_path=sentiment_path)
    
    with open(idf_path, "rb") as f:
        idf_dict = pickle.load(f)
    """テーブル名の宣言"""
    log_table = app.config['LOG_TABLENAME']
    result_table = app.config['RESULT_TABLENAME']
    annotation_table = app.config['ANNOTATION_TABLENAME']
    if test_flag:
        data_table = app.config['TEST_TABLENAME']
    else:
        event_table = app.config['EVENT_TABLENAME']
        kessan_table = app.config['KESSAN_TABLENAME']
        # data_table = app.config['FR2_TABLENAME']

    # app.run(debug=True)
    app.run()