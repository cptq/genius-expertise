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

def lyric_vectorizer(song_to_lyrics):
    vectorizer = TfidfVectorizer(tokenizer=tokenize_lyrics, binary=True, norm=None, smooth_idf=False)
    vectorizer.fit(song_to_lyrics.values())
    return vectorizer
    
def make_tfidf_matrix(song_to_lyrics):
    """Builds tfidf matrix on lyrics"""
    #vectorizer = lyric_vectorizer()
    #tfidf_matrix = vectorizer.fit_transform(song_to_lyrics.values())
    vectorizer = lyric_vectorizer(song_to_lyrics)
    tfidf_matrix = vectorizer.transform(song_to_lyrics.values())
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

def covered_stat(song, song_to_lyrics, song_to_ann_lyrics):
    """ computes coverage of song lyrics by annotations
    
    'song' parameter is the song name
    """
    if song not in song_to_ann_lyrics or song not in song_to_lyrics:
        return 0
    lyric_lst = song_to_ann_lyrics[song]
    lyrics = song_to_lyrics[song]
    lyrics = proc_lyrics(lyrics)
    prev_len = len(lyrics)
    if prev_len == 0:
        return 0
    for t in lyric_lst:
        lyrics = lyrics.replace(t, '')
    covered_val = 1-len(lyrics)/prev_len
    return covered_val

def compute_ann_idx_to_score(song_to_lyrics, annotation_info):
    """ computes a dictionary from annotation index to originality
    """
    print('Computing annotation scores, takes a few minutes')
    vectorizer = lyric_vectorizer(song_to_lyrics)
    ann_idx_to_score = {}
    song_to_annot_idx = collections.defaultdict(list)
    for i, a in enumerate(annotation_info):
        if a['type'] == 'reviewed':
            song_name = a['song']
            song_to_annot_idx[song_name].append(i)
    for s in song_to_annot_idx:
        song_to_annot_idx[s].sort(key=lambda i:
                            to_dt_a(annotation_info[i]['time']).timestamp())
        idx_lst = song_to_annot_idx[s]
        content_lst = [BeautifulSoup(
            annotation_info[i]['edits_lst'][-1]['content'], "lxml").get_text()
                      for i in idx_lst ]
        score_matrix = vectorizer.transform(content_lst)
                                            
        score_matrix.data -= 1
        for tr, i in enumerate(idx_lst):
            if score_matrix[tr, :].nnz:
                _, _, v = scipy.sparse.find(score_matrix[tr,:])
                score = (np.percentile(v, 60) + np.percentile(v, 75) + np.percentile(v, 90))/3
            else:
                score = 0
            ann_idx_to_score[i] = score
    return ann_idx_to_score
