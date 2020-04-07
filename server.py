import os
import sys
import io
import csv
import nltk
import pandas as pd
import numpy as np
import re
import math
import string
import random
import tempfile
import itertools
import ast
import json

from keras import backend as K
from ast import literal_eval
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict, Counter
from nltk.corpus import wordnet as wn
from keras.models import model_from_json
from gensim.models.doc2vec import Doc2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


from flask import Flask, render_template, request, jsonify, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
# rule = request.url_rule

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow import keras
from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath('./model/Lyric Generation/lyricgen_models'))
# from load import *

# init flask apps
app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
# db = SQLAlchemy(app)

global model, seq_l, num_songs, genre_file
SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
check = ['Pop', 'Hip-Hop', 'Rock', 'Metal', 'Country', 'Jazz', 'Electronic', 'Folk', 'R&B', 'Indie']
sim_artist_d2v_path = os.path.join(SITE_ROOT, "model\D2V", "d2v_final.model")

master_csv_path = os.path.join(SITE_ROOT, "model\Lyric Generation\lyric data", "SongLyrics_Final.csv")
master_csv = pd.read_csv(master_csv_path)
master_csv_no_stopwords = pd.read_csv(os.path.join(SITE_ROOT, "model\Lyric Generation\lyric data", "SongLyrics_Final_list.csv"))

d2v_model = Doc2Vec.load(sim_artist_d2v_path)
d2v_path = os.path.join(SITE_ROOT, "model\D2V", "d2v_20artists_3dfg.json")
d2v_graph_path = os.path.join(SITE_ROOT, "model\D2V", "d2v_20artists_active.json")
ag_dict_path = os.path.join(SITE_ROOT, "model\D2V", "ag_dict.json")

ga_list = []

class ga_class:

    def __init__(self, ga_inp):

        self.ga_inp = ga_inp
        self.radar_data = -1
        self.kpidata = -1
        self.clouddata = -1
        self.addga()

    def addga(self):
        ga = self.ga_inp
        self.radar_data = getRadarData([getProperName(ga)])
        self.kpidata = getKPIData([getProperName(ga)])
        self.clouddata = getclouddata([getProperName(ga)])

with open(ag_dict_path) as json_file:
    ag_dict = json.load(json_file)

seq_l = int()
num_songs = int()
genre_file = str()

def getGenre(artist):
  res = 'not found'
  for g, a in ag_dict.items():
    for ele in a:
      if str(ele) == artist:
        # print(res)
        res = g
  return res

def shorten(s):
    return s.translate(str.maketrans('', '', string.punctuation)).lower().replace(" ", "")


def load_model_details(file_str):
  global seq_l, num_songs, genre_file, check

  extract = re.search(r'([^(\\|\/)]*)(\\|\/)*$', file_str)[0]
  seq_l = int(re.search(r'(?<=songs_).*?(?=sl)', extract)[0])
  genre_file = check[list(map(shorten, check)).index(shorten(str(re.search(r'^[^_]+', extract)[0])))]
  num_songs = int(re.search('(?<=_).*?(?=songs_)', extract)[0])

# model = load_model('Indie_1000songs_5sl_100_100_100_vsize.h5')

def init_model(h5_url, json_url):
    json_file = open(json_url)
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    json_file.close()

    # load weights into model
    loaded_model.load_weights(h5_url)

    print("Loaded Model from disk")

    # Compile and evaluate loaded model
    # loaded_model.Compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

    graph = tf.get_default_graph()

    return loaded_model

def Text2Seq(texts, word_to_index):
    indices = np.zeros((1, len(texts)), dtype=int)
    
    for i, text in enumerate(texts):
        indices[:, i] = word_to_index[text]

    return indices

def PaddedSeq(seq, maxlen):
    start = seq.shape[1] - maxlen
    return seq[:, start: start + maxlen]

def GenSeed(genrelist, n_words):
    lyric = ''
    for ix in range(len(genrelist)):
      lyric += genrelist[ix]
    lyric = lyric.lower().replace("\n", ' endline ')
    lyric = ' '.join(lyric.split())
    corpus = re.findall(r"[\w']+", lyric)
    while "" in corpus:
        corpus.remove("")
    vocab = set(corpus)

    start_index = random.randint(0, len(vocab))
    in_text = (' '.join(lyric.split()[start_index:start_index + n_words]))
    # in_text = in_text.translate(str.maketrans('', '', string.punctuation))
    in_text.replace("endline", "endLine")
    in_text = ' '.join(re.findall(r"[\w']+", in_text))

    return in_text

def sample(preds, temperature=1.0):
  # helper function to sample an index from a probability array
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def GenerateLyrics(global_LyricGenInst, n_words, diversity=0.2):

    model = global_LyricGenInst.getmodel()
    word_to_index = global_LyricGenInst.word_to_index
    seq_l = global_LyricGenInst.seq_l
    num_songs = global_LyricGenInst.num_songs
    genrelist = global_LyricGenInst.genrelist
    # model = load_model(gen_filepath)
    index_to_word = {i: w for w, i in word_to_index.items()}

    result = list()
    
    seed = GenSeed(genrelist, n_words*seq_l)
    print('Seed: ' + seed + '\n')
    # result = seed.split(' ')
    # seq_length = len(x[0])
    for _ in range(n_words):
      # print(seed)
      encoded = Text2Seq(seed.split()[1:], word_to_index)
      # print("1: ",encoded)
      encoded = PaddedSeq(encoded, maxlen=seq_l)
      # print("2: ",encoded)

      #-------------------

      preds = model.predict(encoded, verbose = 0)[0]
      next_index = sample(preds, diversity)
      out_word = index_to_word[next_index]

      #-------------------

      # yhat = model.predict_classes(encoded, verbose=0)
      # # print(yhat)
      # out_word = ''
      
      # for word, index in word_to_index.items():
      #     if index in yhat:
      #         out_word = word
      #         break

      #-------------------
      
      seed += ' ' + out_word
      
      # seed = result[-1] + ' ' + out_word
      result.append(out_word)
      out = ' '.join(result) 
    return out

def checkValid(inp):
    # print("Entering `checkValid`")
    out = shorten(inp) in list(map(shorten, check)) + list(map(shorten, list(master_csv.artist.unique())))
    # print("Exiting `checkValid`")
    return out

def isGenre(inp):
    return checkValid(inp) and (shorten(inp) in list(map(shorten, check)))

def isArtist(inp):
    return checkValid(inp) and (not isGenre(inp))

def getProperName(inp):
    temp_master_csv_list = list(master_csv.artist.unique())

    short = shorten(inp)
    if isGenre(short):
        return check[list(map(shorten, check)).index(shorten(short))]
    else:   
        return temp_master_csv_list[list(map(shorten, temp_master_csv_list)).index(shorten(short))]


@app.route('/')
@app.route("/main", methods=["GET", "POST"])
def main():
    global ga_list
    if request.method == 'GET':
        return render_template("main2.html", error = 0)
    ga_list.append(request.form['getsearch'])
    d2vmodel = get3dgfmodel()

    # TODO add top n words functionality
    # top_n
    return redirect(url_for('explore', ga=ga_list, d2vmodel = d2vmodel))

@app.route('/explore', methods=['POST', 'GET'])
def explore():
    global ga_list
    ga = request.form.get('getsearch', 'Error hehe')
    ga_list = []

    if checkValid(ga):
        
        temp_class = ga_class(ga)
        data = getLabelsData(50, getProperName(ga))
        radarData = temp_class.radar_data
        simAD = getSimArtistData(getProperName(ga))
        kpidata = temp_class.kpidata
        clouddata = temp_class.clouddata
        ga_list.append(temp_class)
        
        return render_template('explore2.html', ga=getProperName(ga), data=data, radarData=radarData[0], simAD = simAD, clouddata = str(clouddata), kpidata = kpidata)

    else:
        error = "Genre or Artist not found"
        return render_template('main2.html', error = error)

@app.route('/lyricgen')
def lyricgen():
    return render_template('lyricgen1.html')

@app.route('/explorer')
@app.route('/getSimArtists', methods=['POST', 'GET'])
def getSimArtistData(inp):
    # inp = 'bbking'
    num_artists = 8
    if isArtist(inp):
        similar_artists = d2v_model.docvecs.most_similar(str(getProperName(inp)), topn=num_artists)
        query = 'https://thumbs-prod.si-cdn.com/_oO5E4sOE9Ep-qk_kuJ945_-qo4=/800x600/filters:no_upscale()/https://public-media.si-cdn.com/filer/d5/24/d5243019-e0fc-4b3c-8cdb-48e22f38bff2/istock-183380744.jpg'
        data = {'simname': [i[0] for i in similar_artists], 'simgenre': [getGenre(i[0]) for i in similar_artists], 'simimgpath': query}
    return data

@app.route('/explore')
@app.route('/getRadarData', methods=['POST', 'GET'])
def getRadarData(inp):
    # inp = 'beyonce'
    res = []
    for ele in inp:
        if isArtist(ele):
            data = [d2v_model.docvecs.n_similarity([ele], master_csv[master_csv['genre'] == genre]['artist'].tolist())*100 for genre in check]
            res.append({'artistname': ele, 'labels' : check, 'datasets' : [{'data' : data}]})
        else:
            return 0
    return res

@app.route('/getRadarAddData', methods=['POST'])
def getRadarAddData():
    if request.json['token'] == 1:
        inp = request.json['artistname']
    # inp = 'beyonce'
    if isArtist(inp):
        d2v_model = Doc2Vec.load(sim_artist_d2v_path)
        data = [d2v_model.docvecs.n_similarity([inp], master_csv[master_csv['genre'] == genre]['artist'].tolist())*100 for genre in check]
        return {'artistname': inp, 'labels' : check, 'datasets' : [{'data' : data}]}
    else:
        return False

@app.route('/AddData', methods=['POST','GET'])
def AddData():
    global ga_list
    # inp = request.get_data(as_text=True)
    inp = 'frank-zappa'

    if inp not in [x for x in ga_list if x.ga_inp == inp]:
        temp_class = ga_class(inp)
        ga_list.append(temp_class)
        res = 1
    
    return getvisdata()

@app.route('/DelData', methods=['POST'])
def DelData():
    global ga_list
    inp = request.get_data(as_text=True)
    todel = [x for x in ga_list if x.ga_inp == 'inp']
    if len(todel) > 0:
        res = 1
        ga_list.remove(todel[0])
    else: 
        res = 0
    return res

# self.ga_inp = ga_inp
#         self.radar_data = -1
#         self.kpidata = -1
#         self.addga()

@app.route('/getvisdata', methods=['POST'])
def getvisdata():
    # global ga_list
    ga_namelist = []
    radarlist = []
    kpi_list = []

    for ele in ga_list:
        ga_namelist.append(ele.ga_inp)
        radarlist.append(ele.radar_data)
        kpi_list.append(ele.kpidata)

    kpidata = list(map(sum, zip(*kpi_list)))
    simAD = getSimArtistData(getProperName(ga_namelist[0]))
    clouddata = getclouddata(ga_namelist)
    res = {'ga': ', '.join(ga_namelist), 'data': getLabelsData(50, getProperName(ga_namelist[0])), 'radarData': radarlist, 'simAD': simAD, 'clouddata': str(clouddata), 'kpidata': [int(x) for x in kpidata]}
    return res
    
    # return render_template('explore2.html', ga=', '.join(ga_namelist), data=getLabelsData(50, getProperName(ga)), radarData=radarlist, simAD = simAD, clouddata = str(clouddata), kpidata = kpidata)

@app.route('/explore')
@app.route('/getKPIData', methods=['POST', 'GET'])
def getKPIData(inp):#most_c, inp
    
    # inp = request.get_data(as_text=True)

    ga_csv = master_csv[(master_csv['genre'].isin(list(inp))) | (master_csv['artist'].isin(list(inp)))]

    kpi_data = ga_csv['lyrics'].str.replace('\n', ' ').str.split(' ').sum()
    total_words = len(kpi_data)
    unique_words = len(set(kpi_data))
    total_phrases = len(''.join(ga_csv['lyrics']).split('\n'))
    total_songs = ga_csv['index'].count()
    # res.append([total_words,unique_words, total_phrases, total_songs])

    return [total_words,unique_words, total_phrases, total_songs]

@app.route('/explore')
@app.route('/getLabelsData', methods=['POST', 'GET'])
def getLabelsData(most_c, inp):#most_c, inp
    # most_c = 5
    # inp = "beyonce"
    if isGenre(inp):
        ga_csv = master_csv[master_csv['genre'] == inp]
    else:
        ga_csv = master_csv[master_csv['artist'] == inp]
    
    ga_lyrics = [ast.literal_eval(ele) for ele in ga_csv['lyrics_list'].tolist()]
    temp = Counter(list(itertools.chain.from_iterable(ga_lyrics))).most_common(most_c)
    
    kpi_data = ga_csv['lyrics'].str.replace('\n', ' ').str.split(' ').sum()
    total_words = len(kpi_data)
    unique_words = len(set(kpi_data))
    total_phrases = len(''.join(ga_csv['lyrics']).split('\n'))
    total_songs = ga_csv['index'].count()


    return {'artistname': inp,'labels' : [i[0] for i in temp], 'data' : [i[1] for i in temp]}
    # return str(ga_lyrics)
    
@app.route('/getclouddata', methods=['POST', 'GET'])
def getclouddata(inp):
    # inp = ['beyonce','eminem']
    ga_csv = master_csv[(master_csv['genre'].isin(list(inp))) | (master_csv['artist'].isin(list(inp)))]
    
    ga_lyrics = [ast.literal_eval(ele) for ele in ga_csv['lyrics_list'].tolist()]
    return ' '.join(' '.join(ele) for ele in ga_lyrics) 
    
@app.route("/get3dgfmodel", methods=["GET", "POST"])
def get3dgfmodel():
    with open(d2v_path) as json_file:
        d2v_3dgf_model = json.load(json_file)
    return str(d2v_3dgf_model)

class GraphDataset:
    def __init__(self):
        self.dataset = {"nodes": [], "links": []}
        
    def addNode(self, nid, name, group):
        temp_res = self.dataset
        topush = {'id': str(nid), 'name': str(name), 'group': group}
        temp_res['nodes'].append(topush)
        self.dataset = temp_res
        
    def checkiflink(self, sourceid, targetid):
        links = self.dataset['links']
        res = False
        if any(item for item in links if (((item['source'] == sourceid) & (item['target'] == targetid)) | ((item['source'] == targetid) & (item['target'] == sourceid)))):
            res = True
        return res

    def addLink(self, sourceid, targetid, dist):
        links = self.dataset['links']
        if not self.checkiflink(sourceid, targetid):
            temp_res = self.dataset
            topush = {'source': str(sourceid), 'target': str(targetid), 'distance': dist}
            temp_res['links'].append(topush)
            self.dataset = temp_res

    def printdataset(self):
        print(self.dataset)

    def getDataset(self):
        return self.dataset

def n_most_sim_genre(artist, genre, n):
    res = []
    for i in range(n):
        res.append(d2v_model.docvecs.most_similar_to_given(artist, [x for x in ag_dict[genre] if x not in res]))
    return res

@app.route("/viewD2VG/<string:artist>/", methods=["POST","GET"])
def viewD2VG(artist):
    # artist = 'frank-zappa'

    # if request.method == 'POST':
    # artist = request.get_data(as_text=True)

    # print('ARTIST PRINT:~~~ ', artist)
    graphdata = GraphDataset()

    temp_list = [ag_dict[str(key)] for key in ag_dict.keys()]
    tag_list = [item for sublist in temp_list for item in sublist] + list(ag_dict.keys()) 
    
    # similar2artist = d2v_model.docvecs.most_similar(artist, topn=5)
    # sim = [ele[0] for ele in similar2artist]
    
    vis1_tag = 'Most similar artists to ' + str(artist) + ' from each genre'
    graphdata.addNode(vis1_tag, str(vis1_tag), getGenre(artist))
    for sa in ag_dict.keys():
        graphdata.addNode(str(sa) + '_id', str(sa), sa)
        graphdata.addLink(str(vis1_tag), str(sa) + '_id', 10)
        for sa2 in n_most_sim_genre(artist, sa, 10):
            graphdata.addNode(str(sa2) + '_id', str(sa2), getGenre(sa2))
            graphdata.addLink( str(sa) + '_id',str(sa2) + '_id', 5)

    vis2_tag = 'Top 100 similar artists to ' + str(artist)
    graphdata.addNode(vis2_tag, str(vis1_tag), getGenre(artist))
    similar2artist = d2v_model.docvecs.most_similar(artist, topn=100)
    sim = [ele[0] for ele in similar2artist]
    for sa in sim:
        graphdata.addNode(str(sa) + '_id2', str(sa), sa)
        graphdata.addLink(str(vis2_tag), str(sa) + '_id2', 2.5)

    return render_template('D2VGraph.html', graphpath = graphdata.getDataset())

# app.route('/updatedata')
# def updatedata():


class LyricGenClass:
    def __init__(self, genre):

        self.genre = genre
        self.model_file = "No model file"
        self.word_to_index = "No word to index"
        self.seq_l = -1
        self.num_songs = -1
        self.genrelist = []

        self.LyricGenData()

    def getmodel(self):
        model = tf.keras.models.load_model(self.model_file)
        return model
    
    def LyricGenData(self):
        global SITE_ROOT
        switcher = {
            "Pop": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Pop_500songs_5sl_100_100_100_vsize.h5"),
            "Hip-Hop": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Hip-Hop_1000songs_5sl_100_100_100_vsize.h5"),
            "Rock": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Rock_1000songs_5sl_100_100_100_vsize.h5"),
            "Metal": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Metal_100songs_10sl_Embedding10_LSTM512_Dropout_LSTM512_Dropout_Dense128_Dense2154test.h5"),
            "Country": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Country_1000songs_5sl_100_100_100_vsize.h5"),
            "Jazz": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Jazz_1000songs_5sl_100_100_100_vsize.h5"),
            "Electronic": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Electronic_1000songs_5sl_100_100_100_vsize.h5"),
            "Folk": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Folk_1000songs_5sl_100_100_100_vsize.h5"),
            "R&B": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "R&B_1000songs_5sl_100_100_100_vsize.h5"),
            "Indie": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Indie_1000songs_5sl_100_100_100_vsize.h5")
        }
        genre_file = self.genre
        model_url = switcher.get(genre_file, os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Metal_100songs_10sl_Embedding10_LSTM512_Dropout_LSTM512_Dropout_Dense128_Dense2154test.h5"))
        self.model_file = model_url
        genre_filepath = os.path.join(SITE_ROOT, "model\Lyric Generation\lyric data\Lyrics_by_genre", genre_file + ".csv")

        check = ['Pop', 'Hip-Hop', 'Rock', 'Metal', 'Country', 'Jazz', 'Electronic', 'Folk', 'R&B', 'Indie']
        def shorten(s):
            return s.translate(str.maketrans('', '', string.punctuation)).lower().replace(" ", "")
        extract = re.search(r'([^(\\|\/)]*)(\\|\/)*$', model_url)[0]
        self.seq_l = int(re.search(r'(?<=songs_).*?(?=sl_)', extract)[0])
        self.num_songs = int(re.search('(?<=_).*?(?=songs_)', extract)[0])
        
        artist_master = pd.read_csv(genre_filepath)
        artist = artist_master.dropna(subset = ['lyrics'])[:self.num_songs]
        ArtistSong = list(artist['lyrics'].astype(str))
        self.genrelist = ArtistSong
        text = ''
        for ix in range(len(ArtistSong)):
            text += ArtistSong[ix]
        text = text.lower().replace("\n", ' endline ')
        text = ' '.join(text.split())
        # print('Corpus length in characters:', len(text))
        corpus = re.findall(r"[\w']+", text)
        while "" in corpus:
            corpus.remove("")
        # print('Corpus length in words:', len(corpus))

        vocab = set(corpus)
        self.word_to_index = {w: i for i, w in enumerate(vocab)}

global_LyricGenInst = LyricGenClass("Pop")

@app.route("/LoadLyricGenData", methods=["GET","POST"])
def LoadLyricGenData():
    global global_LyricGenInst
    # genre_file = request.get_data(as_text=True)
    genre_file = "Metal"
    global_LyricGenInst = LyricGenClass(genre_file)
    res = "genre: " + global_LyricGenInst.genre + "\nmodel file: " + global_LyricGenInst.model_file + "\nword to index: " + str(len(global_LyricGenInst.word_to_index)) + "\nseq_l: " + str(global_LyricGenInst.seq_l) + "\nnumsongs: " + str(global_LyricGenInst.num_songs) + "\ngenre list: " + str(len(global_LyricGenInst.genrelist)) 
    return res

@app.route("/LoadnGen", methods=["GET","POST"])
def LoadnGen():
    global model, SITE_ROOT 

    # genre = request.get_data(as_text=True)
    genre = 'Metal'
    # print(genre)
    switcher = {
        "Pop": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Pop_500songs_5sl_100_100_100_vsize.h5"),
        "Hip-Hop": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Hip-Hop_1000songs_5sl_100_100_100_vsize.h5"),
        "Rock": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Rock_1000songs_5sl_100_100_100_vsize.h5"),
        "Metal": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Metal_100songs_10sl_Embedding10_LSTM512_Dropout_LSTM512_Dropout_Dense128_Dense2154test.h5"),
        "Country": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Country_1000songs_5sl_100_100_100_vsize.h5"),
        "Jazz": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Jazz_1000songs_5sl_100_100_100_vsize.h5"),
        "Electronic": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Electronic_1000songs_5sl_100_100_100_vsize.h5"),
        "Folk": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Folk_1000songs_5sl_100_100_100_vsize.h5"),
        "R&B": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "R&B_1000songs_5sl_100_100_100_vsize.h5"),
        "Indie": os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Indie_1000songs_5sl_100_100_100_vsize.h5")
    }

    model_url = switcher.get(genre, os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Metal_100songs_10sl_Embedding10_LSTM512_Dropout_LSTM512_Dropout_Dense128_Dense2154test.h5"))

    # load_model_details(h5_url)
    # json_top_file = re.sub(r'\.(.*)', '.json', re.search(r'([^(\\|\/)]*)(\\|\/)*$', h5_url)[0])
    # json_url = os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", json_top_file)
    # data_f = os.path.join(SITE_ROOT, "model\Lyric Generation\lyric data\Lyrics_by_genre", genre_file + ".csv")
    
    # global_model = tf.keras.models.load_model(model_url)

    # res = "h5: " +  h5_url + "\njson top: " + json_top_file + "\njson: " + json_url + "\ndata: " + data_f

    # print("h5: ", h5_url)
    # print("json top: ", json_top_file)
    # print("json: ", json_url)
    # print("data: ", data_f)
    # print("model: ", model)
    return GenerateLyrics(global_LyricGenInst, 100, 0.6)


@app.route('/generate')
def generate():
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    model_f = os.path.join(SITE_ROOT, "model\Lyric Generation\lyricgen_models", "Country_500songs_5sl_100_100_100_vsize.h5")
    data_f = os.path.join(SITE_ROOT, "static\datasets\Lyric Generation\lyric data\Lyrics_by_genre", "Country.csv")
    model = tf.keras.models.load_model(model_f, compile=False)
    # return GenerateLyrics(data_f, model_f, 100) 

    return model

if __name__ == "__main__":
    app.run(debug = True, host="0.0.0.0", port = 8080)