import MeCab
import re
import sys
import numpy as np
import gensim
import models
import traceback
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity



"""
model_path = "/home/m2018higarashi/Src/causal_network/frnp/data/models/wikipedia/entity_vector.model.txt"
model_path = "/home/m2018higarashi/Src/causal_network/frnp/data/models/wikipedia/entity_vector.model.bin"
dict_path = "/usr/local/lib/mecab/dic/mecab-ipadic-neologd"
"""



"""1. 形態素解析
INPUT: date, risk
OUTPUT: リスクの分散表現
"""
# 全く指定せず，単純に分かち書きをするなら,,,
def wakati(sentence, mecab):
    mecab.parse("")
    #分割して、名刺をcontent_nounにそれぞれ格納
    word_list = []
    node = mecab.parseToNode(sentence)
    while node:
        if node.feature.split(",")[-3]=="*":
            genkei = node.surface
        else:
            genkei = node.feature.split(",")[-3]        
        word_list.append(genkei)
        node = node.next
    return word_list[1:-1]

# 名詞に絞った分かち書きをするなら
def noun_extraction(sentence, mecab):
    # mecab = MeCab.Tagger("-d " + dict_path)
    mecab.parse("")
    #分割して、名刺をcontent_nounにそれぞれ格納
    noun_list = []
    node = mecab.parseToNode(sentence)
    while node:
        if re.search("^(名詞)", node.feature):
            noun_list.append(node.surface)
        node = node.next
    return noun_list

#内容語のみのPICK UP するなら
def word_extraction(sentence, mecab):
    mecab.parse("")
    #分割して、名刺をcontent_nounにそれぞれ格納
    word_list = []
    node = mecab.parseToNode(sentence)
    while node:
        hinshi = node.feature.split(",")[0]
        hinshi_ = node.feature.split(",")[1]
        if node.feature.split(",")[-3]=="*":
            genkei = node.surface
        else:
            genkei = node.feature.split(",")[-3]
        if hinshi =="名詞" and hinshi_ in ["サ変接続", "一般", "固有名詞", "形容動詞語幹", "副詞可能"]:
            word_list.append(genkei)
        elif (hinshi, hinshi_)==("動詞", "自立"):
            word_list.append(genkei)
        elif (hinshi, hinshi_)==("形容詞", "自立"):
            word_list.append(genkei)
        if node.surface == "増":
            word_list.append("増加")
        elif node.surface == "減":
            word_list.append("減少")
        node = node.next
    return word_list

def noun2vec(noun_list, model, idf_dict):
    # 時間かかる
    vecs = np.empty((0,model.vector_size), float)
    idfs = np.array([])
    for noun in noun_list:

        try:
            vec = model.wv[noun].reshape(1, model.vector_size)
        except:
            try:
                vec = model.wv["[" + noun + "]"].reshape(1, model.vector_size)
            except:
                vec = np.zeros((1,model.vector_size))

        
        if noun in idf_dict.keys():
            idf = idf_dict[noun]
        else:
            idf = 0
        vecs = np.append(vecs, vec, axis=0)
        idfs = np.append(idfs, idf)
    if np.count_nonzero(idfs):
        idf_mean = idfs.sum()/np.count_nonzero(idfs)
        idfs[idfs == 0] = idf_mean
    weighted_sum_vec = np.dot(idfs,vecs)
    return weighted_sum_vec

def remove_stopword(word_list, stopwords):
    valid = []
    for word in word_list:
        if word not in stopwords:
            valid.append(word)
    return valid

def sentence2vec(sentence, mecab, model, stopwords, idf_dict):
    # noun_list = noun_extraction(sentence, mecab)
    noun_list = word_extraction(sentence, mecab)
    noun_valid = remove_stopword(noun_list, stopwords)
    vec = noun2vec(noun_valid, model, idf_dict)
    return noun_valid, vec

def show_sentences_similarity(sentence_a, sentence_b, model, stopwords, idf_dict):
    words_a ,vec_a = nlptools.sentence2vec(sentence_a, mecab, model, stopwords, idf_dict)
    words_b ,vec_b = nlptools.sentence2vec(sentence_b, mecab, model, stopwords, idf_dict)
    sim = round(cosine_similarity([vec_a, vec_b])[0][1], 3)
    return sim

def get_sentence_sentiment(noun_list, sentiment_dict):
    sentence_sentiment = 0
    for noun in noun_list:
        if noun in sentiment_dict.keys():
            sentence_sentiment += sentiment_dict[noun]
        else:
            pass
    return sentence_sentiment

def get_sentiment_dict(sentiment_path):
    lines = [line for line in open(sentiment_path, 'r')]  # ファイル読み込み
    sentiment_dict={}
    for line in lines[1:]:
        key = line.split(",")[1]
        value = float(line.split(",")[2])
        sentiment_dict[key] = value
    return sentiment_dict


"""2. 探索対象の選定
INPUT: sector, period, date
OUTPUT: 銘柄コード、記事（探信 or 新聞）の発行日時のdict
"""
def define_search_area(con, sector, from_date, to_date, table_name, stopwords):
    filter_sql = """
    SELECT * FROM {TABLENAME}
    WHERE  {FROMDATE} < `issue_date`
    AND `issue_date` < {TODATE}
    """.format(TABLENAME=table_name, FROMDATE=from_date, TODATE=to_date)
    if stopwords:
        for word in stopwords:
            filter_sql += """
            AND (`result_noun` NOT LIKE "%{WORD}%" AND `cause_noun` NOT LIKE "%{WORD}%")
            """.format(WORD=word)
    if sector == "all":
        pass
    else:
        filter_sql += "AND `sector_code`= {SECTOR}".format(SECTOR=sector)
    l = models.query2list(con, filter_sql)
    return l

def filter_necessary_data(search_area, valid_keys):
    valid_search_dict ={}
    for dic in search_area:
        valid_dic = {}
        for key in valid_keys:
            if "vec" in key:
                str_vec = dic.get(key)
                valid_dic[key] = np.fromstring(str_vec, dtype=float, sep=' ')
            else:
                valid_dic[key] = dic.get(key)
        valid_search_dict[valid_dic["id"]] = (valid_dic)
    return valid_search_dict

"""3. 実際に探索
INPUT: 探索対象dict、Layer_num
OUTPUT: 階層、類似度、文、銘柄、発行日時のdict
"""
def make_init_result(direction, search_dict, thre, init_sentence , init_vec, use_sentiment, init_sentiment, topn):
    result = {}
    from_sentence = init_sentence
    if direction == "downward":
        from_vec = init_vec
        l=[]
        for to_id in tqdm(search_dict.keys()):
            to_vec = search_dict[to_id]["cause_vec"]
            to_sentiment = search_dict[to_id]["cause_sentiment"]
            try:
                sim = round(cosine_similarity([from_vec, to_vec])[0][1], 3)
            except:
                print(traceback.format_exc())    # いつものTracebackが表示される
                print(from_vec, to_vec)

            if use_sentiment:
                if sim > thre and init_sentiment*to_sentiment > 0:
                    to_sentence_cause = search_dict[to_id]["cause"]
                    to_sentence_result = search_dict[to_id]["result"]
                    to_node = to_id + "\n"  + to_sentence_cause + "\n" +to_sentence_result
                    l.append((to_id, to_node, sim))
            else:
                if sim > thre:
                    to_sentence_cause = search_dict[to_id]["cause"]
                    to_sentence_result = search_dict[to_id]["result"]
                    to_node = to_id + "\n" + to_sentence_cause + "\n" +to_sentence_result
                    l.append((to_id, to_node, sim))
        l = filter_topn_result(l, topn)
        result[from_sentence] = l 
        print("{}個の子ノードが見つかりました".format(len(l)))
    
    if direction == "upward":
        from_vec = init_vec
        l=[]
        for to_id in tqdm(search_dict.keys()):
            to_vec = search_dict[to_id]["result_vec"]
            to_sentiment = search_dict[to_id]["result_sentiment"]
            sim = round(cosine_similarity([from_vec, to_vec])[0][1], 3)
            if use_sentiment:
                if sim > thre and init_sentiment*to_sentiment > 0:
                    to_sentence_cause = search_dict[to_id]["cause"]
                    to_sentence_result = search_dict[to_id]["result"]
                    to_node = to_id + "\n" + to_sentence_cause + "\n" +to_sentence_result
                    l.append((to_id, to_node, sim))

            else:
                if sim > thre:
                    to_sentence_cause = search_dict[to_id]["cause"]
                    to_sentence_result = search_dict[to_id]["result"]
                    to_node = to_id + "\n" + to_sentence_cause + "\n" +to_sentence_result
                    l.append((to_id, to_node, sim))
        l = filter_topn_result(l ,topn)
        result[from_sentence] = l
        print("{}個の子ノードが見つかりました".format(len(l)))
    return result


def search_causeal_relations(direction, search_dict, thre, old_result, use_sentiment, topn):
    new_result={}
    if direction == "downward":
        for res_list in list(old_result.values()):
            for from_id, from_node, sim in res_list:
                from_vec = search_dict[from_id]["result_vec"] #
                from_sentiment = search_dict[from_id]["result_sentiment"]
                print("searching for ",from_node, "--> ??")
                l=[]
                for to_id in tqdm(search_dict.keys()):
                    to_vec = search_dict[to_id]["cause_vec"] #
                    to_sentiment = search_dict[from_id]["cause_sentiment"]
                    sim = round(cosine_similarity([from_vec, to_vec])[0][1], 3)
                    if use_sentiment:
                        if sim > thre and from_sentiment*to_sentiment > 0:
                            to_sentence_cause = search_dict[to_id]["cause"]
                            to_sentence_result = search_dict[to_id]["result"]
                            to_node = to_id + "\n" + to_sentence_cause + "\n" +to_sentence_result
                            l.append((to_id, to_node, sim))
                    else:
                        if sim > thre:
                            to_sentence_cause = search_dict[to_id]["cause"]
                            to_sentence_result = search_dict[to_id]["result"]
                            to_node = to_id + "\n" + to_sentence_cause + "\n" +to_sentence_result
                            l.append((to_id, to_node, sim))
                l = filter_topn_result(l ,topn)
                new_result[from_node] = l
                print("{}個の子ノードが見つかりました".format(len(l)))
                
    if direction == "upward":
        for res_list in list(old_result.values()):
            for from_id, from_node, sim in res_list:
                from_vec = search_dict[from_id]["cause_vec"] #
                from_sentiment = search_dict[from_id]["cause_sentiment"]
                l=[]
                for to_id in tqdm(search_dict.keys()):
                    to_vec = search_dict[to_id]["result_vec"] #
                    to_sentiment = search_dict[from_id]["result_sentiment"]
                    sim = round(cosine_similarity([from_vec, to_vec])[0][1], 3)
                    if use_sentiment:
                        if sim > thre and from_sentiment*to_sentiment > 0:
                            to_sentence_cause = search_dict[to_id]["cause"]
                            to_sentence_result = search_dict[to_id]["result"]
                            to_node = to_id + "\n" + to_sentence_cause + "\n" +to_sentence_result
                            l.append((to_id, to_node, sim))
                    else:
                        if sim > thre:
                            to_sentence_cause = search_dict[to_id]["cause"]
                            to_sentence_result = search_dict[to_id]["result"]
                            to_node = to_id + "\n" + to_sentence_cause + "\n" +to_sentence_result
                            l.append((to_id, to_node, sim))
                l = filter_topn_result(l ,topn)
                new_result[from_node] = l
                print("{}個の子ノードが見つかりました".format(len(l)))
    return new_result



def show_2sentences(search_dict, direction, id1, id2):   
    if direction == "downward":
        sentence1 = search_dict[id1]["result"]
        sentence2 = search_dict[id2]["cause"]
        
    if direction == "upward":
        sentence1 = search_dict[id1]["cause"]
        sentence2 = search_dict[id2]["result"]
    return sentence1, sentence2 


def show_sentence(search_dict, id, event):
    return search_dict[id][event]

def result_length(result):
    length=0
    for key in result.keys():
        length += len(result[key])
    return length

def get_node_edge(results, init_sentence, direction):
    node_list = []
    edge_list = []
    node_list.append(init_sentence)
    layer = 1
    if direction == "downward":
        for result in results:
            for from_node in list(result.keys()):
                for to_id, to_node, sim in result[from_node]:
                    edge = [layer, from_node, to_node, sim]
                    edge_list.append(edge)
                    node_list.append(to_node)
            layer += 1

    if direction == "upward":
        for result in results:
            for to_node in list(result.keys()):
                for from_id, from_node, sim in result[to_node]:
                    edge = [layer, from_node, to_node, sim]
                    edge_list.append(edge)
                    node_list.append(from_node)
            layer += 1
    

    # set_node_list=list(set(node_list))
    # str_edge_list =[]
    # for ele_list in edge_list:
    #     maped_list = map(str, ele_list)  #mapで要素すべてを文字列に
    #     mojiretu = ','.join(maped_list)
    #     str_edge_list.append(mojiretu)
    # str_edge_list=list(set(str_edge_list))
    # set_edge_list = [s.split(",") for s in str_edge_list]
        
    # return set_node_list, set_edge_list
    return node_list, edge_list

def filter_topn_result(l, topn):
    if len(l)>topn:
        sims = [tpl[2] for tpl in l]
        topn_index = sorted(range(len(sims)), key=lambda i: sims[i])[-topn:]
        filtered_l = [l[i] for i in topn_index]
    else:
        filtered_l = l
    return filtered_l
"""
1-3をLayer_numだけループする
"""

if __name__ == "__main__":
    args = sys.argv
    sentence = args[1]
    dict_path = args[2]
    model_path = args[3]
    mecab = MeCab.Tagger("-d " + dict_path)
    model = gensim.models.word2vec.KeyedVectors.load_word2vec_format(model_path)
    sentence2vec(sentence, mecab, model)
