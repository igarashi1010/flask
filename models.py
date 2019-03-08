"""
ビジネスロジックモジュール
"""
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd
import numpy as np
import time
import pickle
import io
import nlptools
import networkx as nx
from graphviz import Digraph
import urllib
from matplotlib.backends.backend_agg import FigureCanvasAgg

def get_relations(con, mecab, model, data_dict, event_tablename, kessan_tablename, topn):

    # date = data_dict["Date"]
    layer_num = data_dict['LayerNum']
    sector = data_dict['Sector']
    risk = data_dict['RiskSentence']
    from_date = data_dict['FromDate']
    to_date = data_dict['ToDate']
    direction = data_dict['Direction']
    thre = data_dict['Thre']
    use_sentiment = data_dict['UseSentiment']

    sentiment_dict = data_dict['SentimentDict']
    test_flag = data_dict['TestFlag']
    idf_dict = data_dict['IdfDict']
    stopwords = data_dict['Stopwords']
    """1. 形態素解析
    INPUT: risk
    OUTPUT: リスクの分散表現
    """
    if test_flag:
        test_vec_path = "test_vec.pkl"
        risk_nouns = ["増収","株価"]
        with open(test_vec_path, "rb") as f:
            risk_vec = pickle.load(f)
    else:
        risk_nouns, risk_vec = nlptools.sentence2vec(sentence=risk, mecab=mecab, model=model, stopwords=stopwords, idf_dict=idf_dict)
    
    risk_sentiment = nlptools.get_sentence_sentiment(noun_list=risk_nouns, sentiment_dict=sentiment_dict)
    """2. 探索対象の選定
    INPUT: sector, period, date
    OUTPUT: 銘柄コード、記事（探信 or 新聞）の発行日時のdict
    """
    print("探索対象を探しています..........")

    event_search_area = nlptools.define_search_area(con=con, sector=sector, from_date=from_date, to_date=to_date, 
        table_name=event_tablename, stopwords=stopwords)

    kessan_search_area = nlptools.define_search_area(con=con, sector=sector, from_date=from_date, to_date=to_date, 
        table_name=kessan_tablename, stopwords=stopwords)

    valid_keys=["id", "cause", "result", "cause_vec", "result_vec", "cause_sentiment", "result_sentiment"]
    event_valid_search_dict = nlptools.filter_necessary_data(event_search_area, valid_keys)
    kessan_valid_search_dict = nlptools.filter_necessary_data(kessan_search_area, valid_keys)

    """3. 実際に探索
    INPUT: 探索対象dict、Layer_num
    OUTPUT: 階層、類似度、文、銘柄、発行日時のdict
    """

    print("探索開始: ", risk)
    print("Making initial result")
    # results = []
    # initial_result = nlptools.make_init_result(direction=direction, search_dict=valid_search_dict, 
    #     thre=thre, init_sentence=risk, init_vec=risk_vec, use_sentiment=use_sentiment, init_sentiment=risk_sentiment, topn=topn)
    # results.append(initial_result)
    # """
    # 1-3をLayer_numだけループする
    # """
    # result = initial_result
    # result_count=len(result.keys())
    # print("1st result: {}".format(result_count))
    # for i in range(1, layer_num):
    #     layer = i+1
    #     print("layer_{} is creating".format(layer))
    #     result = result = nlptools.search_causeal_relations(direction=direction, search_dict=valid_search_dict, thre=thre, old_result=result,
    #             use_sentiment=use_sentiment, topn=topn)
    #     result_count=len(result.keys())
    #     print("result: {}".format(result_count))
    #     results.append(result)
    #     # while result.values():
    #     #     print("layer_{} is creating".format(layer))
    #     #     result = nlptools.search_causeal_relations(direction=direction, search_dict=valid_search_dict, thre=thre, old_result=result,
    #     #         use_sentiment=use_sentiment)
    results = []
    if layer_num == 1:
        print("risk-->event")
        result = nlptools.make_init_result(direction=direction, search_dict=kessan_valid_search_dict, 
        thre=thre, init_sentence=risk, init_vec=risk_vec, use_sentiment=use_sentiment, init_sentiment=risk_sentiment, topn=topn)
        results.append(result)
    else:
        print("risk-->event")
        initial_result = nlptools.make_init_result(direction=direction, search_dict=event_valid_search_dict, 
            thre=thre, init_sentence=risk, init_vec=risk_vec, use_sentiment=use_sentiment, init_sentiment=risk_sentiment, topn=topn)
        results.append(initial_result)
        result = initial_result
        for i in range(layer_num-2):
            print("event-->event")
            result = nlptools.search_causeal_relations(direction=direction, search_dict=event_valid_search_dict, thre=thre, old_result=result,
                use_sentiment=use_sentiment, topn=topn)
            results.append(result)
        print("event-->kessan")
        result = nlptools.search_causeal_relations(direction=direction, search_dict=kessan_valid_search_dict, thre=thre, old_result=result,
                use_sentiment=use_sentiment, topn=topn)
        results.append(result)
    node_list, edge_list = nlptools.get_node_edge(results=results, init_sentence=risk, direction=direction)

    """
     *センチメント情報の獲得*
     """    
    node_sentiment_dict, node_sentiment_pair_dict = get_node_sentiment(con, tablename, node_list)
    return node_list, edge_list, valid_search_dict, risk_sentiment, node_sentiment_dict, node_sentiment_pair_dict

def query2list(con, query):
    cur = con.cursor(dictionary=True)
    cur.execute(query)
    l = cur.fetchall()
    cur.close()
    return l

def get_sectors(con):
    cur = con.cursor()
    sector_sql="""
    SELECT * FROM `sector_definition`;
    """
    l = query2list(con, sector_sql)
    return l

def insert_log(con, TABLENAME, date, layer_num, sector, risk, from_date, to_date, search_result_str, thre, direction, risk_sentiment, use_sentiment):
    """ INSERT処理 """
    insert_sql = """
    insert into {TABLENAME} (AnalyzeDate, LayerNum, Sector, RiskSentence, FromDate, ToDate, SearchResult, Threshold, Direction, Sentiment, UseSentiment) 
    values ({AnalyzeDate}, "{LayerNum}", "{Sector}", "{RiskSentence}", {FromDate}, {ToDate}, "{SearchResult}", "{Threshold}", "{Direction}", {Sentiment},{UseSentiment})
    """.format(TABLENAME=TABLENAME, AnalyzeDate=date, LayerNum=layer_num, Sector=sector,
    RiskSentence=risk, FromDate=from_date, ToDate=to_date, SearchResult=search_result_str, 
    Threshold=thre, Direction=direction, Sentiment=risk_sentiment, UseSentiment=use_sentiment)
    cur = con.cursor()
    # print(insert_sql)
    cur.execute(insert_sql)
    ID = cur.lastrowid
    con.commit()
    return ID

def insert_result(con, TABLENAME, ID, edge_list, date):
    if edge_list:
        insert_column = """
        insert into {TABLENAME} (RiskID, AnalyzeDate, Layer, FromNode, ToNode, Similarity) values
        """.format(TABLENAME=TABLENAME)

        insert_datas = []
        for edge in edge_list:
            layer = edge[0]
            from_node = edge[1]
            to_node = edge[2]
            sim = edge[3]
            data = """
            ({RiskID}, {AnalyzeDate}, {Layer}, '{FromNode}', '{ToNode}', {Similarity})
            """.format(RiskID=ID, AnalyzeDate=date, Layer=layer, FromNode=from_node, ToNode=to_node, Similarity=sim)
            insert_datas.append(data)

        s = ','.join(insert_datas)
        insert_sql = insert_column + s + ";"
        cur = con.cursor()
        cur.execute(insert_sql)
        con.commit()

def delete(con, TABLENAME ,ID):
    """ 結果削除処理 """
    delete_sql = """
    DELETE FROM {TABLENAME} 
    WHERE `RiskID`={ID};""".format(TABLENAME=TABLENAME, ID=ID)
    cur = con.cursor()
    cur.execute(delete_sql)
    con.commit()

def insert_annotate(con, TABLENAME, edge_ID, res):
    cur = con.cursor()
    insert_sql = """
    INSERT INTO {TABLENAME} (EdgeID, Score) values({EdgeID}, {Score});
    """.format(TABLENAME=TABLENAME, Score=res, EdgeID=edge_ID)
    cur.execute(insert_sql)
    con.commit()

def get_data(con, TABLENAME ,ID):
    get_sql = """
    SELECT * FROM {TABLENAME}
    WHERE `RiskID`={ID};""".format(TABLENAME=TABLENAME, ID=ID)
    result = query2list(con, get_sql)
    return result

def draw_network(edge_list, event_list, spring_k, search_dict, ID):
    #matplotlibで描画-->保存
    fig = plt.figure(figsize=(40, 40))
    G = nx.DiGraph()
    # 重み付きのファイルの読み込み
    for i in edge_list+event_list:
        G.add_edge(i[0], i[1], weight=i[2])

    # レイアウトと頂点の色を適当に設定
    pos = nx.spring_layout(G, k=spring_k)

    # ラベルの設定
    nx.draw_networkx_labels(G,pos,font_family='IPAexGothic', font_size=15)
    edge_labels = {(i, j): w['weight'] for i, j, w in G.edges(data=True)}

    # ノードの色の設定
    valid_cause=[d["cause"] for d in search_dict.values()]
    valid_result=[d["result"] for d in search_dict.values()]
    node_color=["r"  if i in valid_cause else "b" if i in valid_result else "y" for i in G.nodes]
    
    # グラフ描画
    nx.draw_networkx_edge_labels(G,pos,font_size=15,edge_labels=edge_labels)
    nx.draw_networkx(G, pos, with_labels=False, node_shape="s", node_color=node_color,
                     alpha=0.5, width =[k["weight"] for i, j, k in G.edges(data=True)])
    plt.axis("off")

    # ファイル名
    filename = "{ID}.png".format(ID=ID)
    # 保存先のパス
    save_path = "./static/results/" + filename

    # # 保存処理を行う
    plt.savefig(save_path)
    # # pltをclose
    plt.close()

def get_HSV(sentiment, weight):
    if sentiment>0:
        H = "1.0"
    else:
        H = "0.667"
    S = str(weight)
    V = "1.0"
    HSV = H + " " + S + " " + V
    return HSV

def get_node_sentiment(con, table_name, node_list):
    id_list = [node.split("\n")[0] for node in node_list]
    sentiment_dict={}
    sentiment_pair_dict={}
    sql = "SELECT `id`, `cause_sentiment`,`result_sentiment` FROM {TABLENAME} WHERE id = '{ID}' \n".format(TABLENAME=table_name, ID=id_list[0])
    for id in id_list[1:]:
        condition =  "OR id='{ID}' \n".format(ID=id)
        sql += condition
    l =  query2list(con, sql)
    for dic in l:
        id = dic["id"]
        node_sentiment = dic["cause_sentiment"] +  dic["result_sentiment"]
        node_sentiment_pair = [dic["cause_sentiment"], dic["result_sentiment"]]
        sentiment_dict[id] = node_sentiment
        sentiment_pair_dict[id] = node_sentiment_pair
    return sentiment_dict, sentiment_pair_dict

def get_node_weight(edge_list):
    
    sim =float(edge_list[0][2])
    pass


def draw_graphviz_colored(node_list, edge_list, init_sentence, init_sentiment, direction, RiskID, sentiment_dict):
    G = Digraph(format="png")
    G.attr("node", shape="box", color="black", fontsize="8")
    G.attr("edge", fontsize="8")
    for node in node_list:
        node_id = node.split("\n")[0]
        
        if node_id==init_sentence:
            G.node(node, color="black")
            
        else:
            node_sentiment = sentiment_dict[node_id]
            node_color = get_HSV(node_sentiment, 1)
            # print(node_color)
            G.node(node, color=node_color)    

    if direction=="downward":
        G.attr("graph", rankdir = "LR")
    elif direction=="upward":
        G.attr("graph", rankdir = "RL")
    for edge in edge_list:
            layer = edge[0]
            from_node = edge[1]
            to_node = edge[2]
            sim = edge[3]
            G.edge(from_node, to_node, label=sim)
    
    # 保存先のパス
    save_path = "./static/results/{}".format(RiskID)
    G.render(save_path)

def draw_graphviz(node_list, edge_list, init_sentence, init_sentiment, direction, RiskID, risk, sentiment_pair_dict):
    G = Digraph(format="png")
    G.attr("node", shape="box", color="black", fontsize="8")
    G.attr("edge", fontsize="8")
    for edge in edge_list:
        layer = edge[0]
        from_node = edge[1]
        to_node = edge[2]
        sim = edge[3]
        from_id = from_node.split("\n")[0]
        to_id = to_node.split("\n")[0]

        if direction=="downward":
            G.attr("graph", rankdir = "LR")
            if from_id==init_sentence:
                edge_sentiment = init_sentiment * sentiment_pair_dict[to_id][0]
            else:
                edge_sentiment = sentiment_pair_dict[from_id][1] * sentiment_pair_dict[to_id][0]
        elif direction=="upward":
            G.attr("graph", rankdir = "RL")
            if to_id==init_sentence:
                edge_sentiment = sentiment_pair_dict[from_id][1] * init_sentiment
            else:
                edge_sentiment = sentiment_pair_dict[from_id][1] * sentiment_pair_dict[to_id][0]
        edge_color = get_HSV(edge_sentiment, sim)
        G.edge(from_node, to_node, label=str(sim), color=edge_color)

    for node in node_list:
        node_id = node.split("\n")[0]
        G.node(node, color="black")
        # if node_id==init_sentence:
        #     G.node(node, color="black")
            
        # else:
        #     node_sentiment = sentiment_dict[node_id]
        #     node_color = get_HSV(node_sentiment, 1)
        #     # print(node_color)
        #     G.node(node, color=node_color)    

    # 保存先のパス
    save_path = "./static/results/{}_{}".format(RiskID, risk)
    G.render(save_path)

