import collections
from helpers import *
import re
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
import numpy as np
from load_data import lyrics_info_gen

# needed for fast coverage computation
parts = ["Intro", "Outro", "Chorus", "Hook",
            "Pre-Hook", "Bridge", "Verse", "Refrain",
            "Pre-Chorus", "Part", "Post-Chorus", 'Interlude']
re_parts = "|".join(map(lambda s: r"\b" + s + r"\b", parts)) # for regex below
regex_parts = re.compile(r"\[("+re_parts+").*\]") # removes part headers and whitespace


def proc_lyrics(lyrics):
    """Process lyrics"""
    lyrics = re.sub(regex_parts, '', lyrics) # remove headers
    lyrics = lyrics.lower() # to lowercase
    lyrics = re.sub('\s', '', lyrics) # remove whitespace
    lyrics = re.sub('\W', '', lyrics) # remove punctuation
    return lyrics

def make_song_to_lyrics(annotation_info):
    """Dictionaries for song to lyrics and song to annotated lyrics"""
    song_to_lyrics = {s['song']: s['lyrics'] for s in lyrics_info_gen()}
    song_to_ann_lyrics = collections.defaultdict(list)
    for ann in sorted(annotation_info, key = lambda a: to_dt_a(a['time']).timestamp()):
        lyrics = proc_lyrics(ann['lyrics'])
        song_to_ann_lyrics[ann['song']].append(lyrics)
    return song_to_lyrics, song_to_ann_lyrics

def tokenize_lyrics(lyrics):
    """Tokenizer for lyrics of entire songs or lyric segments of annotations"""
    lyrics = re.sub(regex_parts, '', lyrics)
    # get rid of punctuation including special ’, –, —
    puncts = string.punctuation + '’' + '–' + '—'
    table = str.maketrans({c:None for c in puncts})
    lyrics = lyrics.translate(table)
    lyrics = lyrics.lower()
    tokens = nltk.word_tokenize(lyrics)
    return tokens

def make_tfidf_matrix(song_to_lyrics):
    """Builds tfidf matrix on lyrics"""
    vectorizer = TfidfVectorizer(tokenizer=tokenize_lyrics, binary=True,
                                     norm=None, smooth_idf=False)
    tfidf_matrix = vectorizer.fit_transform(song_to_lyrics.values())
    tfidf_matrix.data -= 1 # subtract 1 to agree with paper definition
    return tfidf_matrix

def compute_song_to_score(song_to_lyrics):
    """Dictionary from song to novelty score"""
    print('Computing song novelty scores, takes a few minutes')
    tfidf_matrix = make_tfidf_matrix(song_to_lyrics)
    song_to_score = {}
    for i, s in enumerate(song_to_lyrics):
        if tfidf_matrix[i,:].nnz:
            _, _, v = scipy.sparse.find(tfidf_matrix[i,:])
            score = (np.percentile(v, 60) + np.percentile(v, 75) + np.percentile(v, 90))/3
        else:
            score = 0
        song_to_score[s] = score
    return song_to_score
