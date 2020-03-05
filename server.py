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
import tensorflow as tf
import tensorflow.compat.v1 as tensf
tensf.disable_v2_behavior()

from keras import backend as K 
from keras.models import load_model
from ast import literal_eval
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict, Counter
from nltk.corpus import wordnet as wn
from keras.models import model_from_json
from gensim.models.doc2vec import Doc2Vec

from flask import Flask, render_template, request, jsonify, url_for, redirect
from flask_sqlalchemy import SQLAlchemy

sys.path.append(os.path.abspath('./model/Lyric Generation/json_h5'))
# from load import *

# init flask apps
app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
# db = SQLAlchemy(app)

global model, seq_l, num_songs, genre_file
SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
check = ['Pop', 'Hip-Hop', 'Rock', 'Metal', 'Country', 'Jazz', 'Electronic', 'Folk', 'R&B', 'Indie']
sim_artist_d2v_path = os.path.join(SITE_ROOT, "model\D2V", "d2v_final.model")
d2v_model = Doc2Vec.load(sim_artist_d2v_path)
master_csv_path = os.path.join(SITE_ROOT, "model\Lyric Generation\lyric data", "SongLyrics_Final.csv")
master_csv = pd.read_csv(master_csv_path)
master_csv_no_stopwords = pd.read_csv(os.path.join(SITE_ROOT, "model\Lyric Generation\lyric data", "SongLyrics_Final_list.csv"))
d2v_path = os.path.join(SITE_ROOT, "model\D2V", "d2v_20artists_3dfg.json")

with open(d2v_path) as json_file:
    d2v_3dgf_model = json.load(json_file)

seq_l = int()
num_songs = int()
genre_file = str()
class artist_by_genre(dict): 
  
    # __init__ function 
    def __init__(self): 
        self = dict() 
          
    # Function to add key:value 
    def add(self, key, value): 
        self[key] = value 
  
# Main Function 
ag_dict = artist_by_genre() 

genre_list = master_csv.genre.unique().tolist()
artist_list = []
for genre in genre_list:
  df_genre = master_csv[master_csv['genre'] == str(genre)]
  top_artists = df_genre.groupby('artist').size().sort_values(ascending = False).keys().tolist()

  temp_list = [artist for artist in top_artists]#[:no_of_artist]
  artist_list.append(temp_list)
  ag_dict.add(str(genre), temp_list)

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

def GenSeed(ArtistSong, n_words):
    lyric = ''
    for ix in range(len(ArtistSong)):
      lyric += ArtistSong[ix]
    lyric = lyric.lower().replace("\n", ' endline ')
    lyric = ' '.join(lyric.split())
    corpus = re.findall(r"[\w']+", lyric)
    while "" in corpus:
        corpus.remove("")
    vocab = set(corpus)

    start_index=random.randint(0, len(vocab))
    in_text = (' '.join(lyric.split()[start_index:start_index + n_words]))
    # in_text = in_text.translate(str.maketrans('', '', string.punctuation))
    in_text.replace("endline", "endLine")
    in_text = ' '.join(re.findall(r"[\w']+", in_text))

    return in_text

def GenerateLyrics(data_file, model, n_words):
    global seq_l, num_songs, genre_file    
    artist = pd.read_csv(data_file)
    ArtistSong = list(artist['lyrics'][:num_songs].astype(str))
    
    # -------------------------

    text = ''
    for ix in range(len(ArtistSong)):
        text += ArtistSong[ix]
    text = text.lower().replace("\n", ' endline ')
    text = ' '.join(text.split())
    print('Corpus length in characters:', len(text))
    corpus = re.findall(r"[\w']+", text)
    while "" in corpus:
        corpus.remove("")
    print('Corpus length in words:', len(corpus))
    
    # -------------------------

    word_to_index = {w: i for i, w in enumerate(set(corpus))}

    # -------------------------

    # model = load_model(model_file)

    # -------------------------

    result = list()
    
    in_text = GenSeed(ArtistSong, 100)
    seq_length = seq_l

    for _ in range(n_words):
        encoded = Text2Seq(in_text.split()[1:], word_to_index)

        encoded = PaddedSeq(encoded, maxlen=seq_length)

        yhat = model.predict_classes(encoded, verbose=0)
        out_word = ''
        
        for word, index in word_to_index.items():
            if index == yhat:
                out_word = word
                break
        
        in_text += ' ' + out_word
        result.append(out_word)

    result = ' '.join(result)    
    return result.replace('endline', '\n')

def checkValid(inp):
    print("Entering `checkValid`")
    out = shorten(inp) in list(map(shorten, check)) + list(map(shorten, list(master_csv.artist.unique())))
    print("Exiting `checkValid`")
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
    if request.method == 'GET':
        return render_template("main2.html", error = 0)
    ga = request.form['getsearch']
    d2vmodel = get3dgfmodel()

    # TODO add top n words functionality
    # top_n
    return redirect(url_for('explore', ga=ga, d2vmodel = d2vmodel))

@app.route('/explore', methods=['POST', 'GET'])
def explore():
    ga = request.form.get('getsearch', 'Error hehe')
        
    if checkValid(ga):

        data = getLabelsData(100, getProperName(ga))
        radarData = getRadarData(getProperName(ga))
        simAD = getSimArtistData(getProperName(ga))
        return render_template('explore.html', ga=getProperName(ga), data=data, radarData=radarData, simAD = simAD)

    else:
        error = "Genre or Artist not found"
        return render_template('main2.html', error=error)

@app.route('/explorer')
@app.route('/getSimArtists', methods=['POST', 'GET'])
def getSimArtistData(inp):
    # inp = 'bbking'
    num_artists = 5
    if isArtist(inp):
        similar_artists = d2v_model.docvecs.most_similar(str(getProperName(inp)), topn=num_artists)
        query = 'https://thumbs-prod.si-cdn.com/_oO5E4sOE9Ep-qk_kuJ945_-qo4=/800x600/filters:no_upscale()/https://public-media.si-cdn.com/filer/d5/24/d5243019-e0fc-4b3c-8cdb-48e22f38bff2/istock-183380744.jpg'
        data = {'simname': [i[0] for i in similar_artists], 'simgenre': [getGenre(i[0]) for i in similar_artists], 'simimgpath': query}
        simartist_html = '';
        for i in range(num_artists):
            simartist_html += '<div class="card" style="display: inline-block;">'
            simartist_html += '<img src="'+ data['simimgpath'] + '" class="card-img-top" alt="">'
            simartist_html += '<div class="card-body">'
            simartist_html += '<h5 class="card-title">' + data['simname'][i] + '</h5>'
            simartist_html += '<p class="card-text">' + data['simgenre'][i] + '</p>'
            simartist_html += '</div>'
            simartist_html += '</div>'
    return data

@app.route('/explore')
@app.route('/getRadarData', methods=['POST', 'GET'])
def getRadarData(inp):
    # inp = 'beyonce'
    if isArtist(inp):
        d2v_model = Doc2Vec.load(sim_artist_d2v_path)
        data = [d2v_model.docvecs.n_similarity([inp], master_csv[master_csv['genre'] == genre]['artist'].tolist())*100 for genre in check]
        return {'artistname': inp, 'labels' : check, 'datasets' : [{'data' : data}]}
    else:
        return False

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

@app.route('/explore')
@app.route('/getLabelsData', methods=['POST', 'GET'])
def getLabelsData(most_c, inp):

    if isGenre(inp):
        genre_csv = master_csv_no_stopwords[master_csv_no_stopwords['genre'] == getProperName(inp)]
    else:
        genre_csv = master_csv_no_stopwords[master_csv_no_stopwords['artist'] == getProperName(inp)]
    
    genre = [ast.literal_eval(ele) for ele in genre_csv['lyrics'].tolist()]
    temp = Counter(list(itertools.chain.from_iterable(genre))).most_common(most_c)
    
    return {'artistname': inp,'labels' : [i[0] for i in temp], 'data' : [i[1] for i in temp]}

@app.route("/get3dgfmodel", methods=["GET", "POST"])
def get3dgfmodel():
    return str(d2v_3dgf_model)

@app.route("/loadmodel/", methods=["GET", "POST"])
def loadmodel():
    global model, SITE_ROOT

    genre = request.get_data(as_text=True)
    # print(genre)
    switcher = {
        "Pop": os.path.join(SITE_ROOT, "model\Lyric Generation\json_h5", "Pop_1000songs_5sl_100_100_100_vsize.h5"),
        "Hip-Hop": os.path.join(SITE_ROOT, "model\Lyric Generation\json_h5", "Hip-Hop_1000songs_5sl_100_100_100_vsize.h5"),
        "Rock": os.path.join(SITE_ROOT, "model\Lyric Generation\json_h5", "Rock_1000songs_5sl_100_100_100_vsize.h5"),
        "Metal": os.path.join(SITE_ROOT, "model\Lyric Generation\json_h5", "Metal_500songs_5sl_100_100_100_vsize.h5"),
        "Country": os.path.join(SITE_ROOT, "model\Lyric Generation\json_h5", "Country_1000songs_5sl_100_100_100_vsize.h5"),
        "Jazz": os.path.join(SITE_ROOT, "model\Lyric Generation\json_h5", "Jazz_1000songs_5sl_100_100_100_vsize.h5"),
        "Electronic": os.path.join(SITE_ROOT, "model\Lyric Generation\json_h5", "Electronic_1000songs_5sl_100_100_100_vsize.h5"),
        "Folk": os.path.join(SITE_ROOT, "model\Lyric Generation\json_h5", "Folk_1000songs_5sl_100_100_100_vsize.h5"),
        "R&B": os.path.join(SITE_ROOT, "model\Lyric Generation\json_h5", "R&B_1000songs_5sl_100_100_100_vsize.h5"),
        "Indie": os.path.join(SITE_ROOT, "model\Lyric Generation\json_h5", "Indie_1000songs_5sl_100_100_100_vsize.h5")
    }

    h5_url = switcher.get(genre, os.path.join(SITE_ROOT, "model\Lyric Generation\json_h5", "Indie_1000songs_5sl_100_100_100_vsize.h5"))
    load_model_details(h5_url)
    json_top_file = re.sub(r'\.(.*)', '.json', re.search(r'([^(\\|\/)]*)(\\|\/)*$', h5_url)[0])
    json_url = os.path.join(SITE_ROOT, "model\Lyric Generation\json_h5", json_top_file)
    data_f = os.path.join(SITE_ROOT, "static\datasets\Lyric Generation\lyric data\Lyrics_by_genre", genre_file + ".csv")
    K.clear_session()
    model = init_model(str(h5_url), str(json_url))
    # model = tf.keras.models.load_model(h5_url)

    res = "h5: " +  h5_url + "\njson top: " + json_top_file + "\njson: " + json_url + "\ndata: " + data_f

    print("h5: ", h5_url)
    print("json top: ", json_top_file)
    print("json: ", json_url)
    print("data: ", data_f)
    print("model: ", model)

    return res

@app.route('/generate')
def generate():
    SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
    model_f = os.path.join(SITE_ROOT, "model\Lyric Generation\json_h5", "Country_500songs_5sl_100_100_100_vsize.h5")
    data_f = os.path.join(SITE_ROOT, "static\datasets\Lyric Generation\lyric data\Lyrics_by_genre", "Country.csv")
    model = tf.keras.models.load_model(model_f, compile=False)
    # return GenerateLyrics(data_f, model_f, 100) 

    return model

if __name__ == "__main__":
    app.run(debug = True, host="0.0.0.0", port = 8080) 