from constants import DATAPATH
import json
import csv
import networkx as nx
        
def load_graph(positive_iq=True):
    """ Loads directed user graph.

        If positive_iq, then only returns subgraph with users of positive iq.
    """
    map_artist_names = {}
    with open(f'{DATAPATH}/map_artist_names.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            map_artist_names[row[0]] = row[1]

    with open(f"{DATAPATH}/user_info.jl", 'r') as f:
        user_info = f.readlines()
    user_info = list(map(json.loads, user_info))
    G = nx.DiGraph()
    G.add_nodes_from(map(lambda u: u["url_name"], user_info))
    
    with open(f"{DATAPATH}/follows.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            row[1] = row[1][1:]
            if row[1] in map_artist_names:
                row[1] = map_artist_names[row[1]]
            if row[0] != row[1]: G.add_edge(row[0], row[1])
    if positive_iq:
        G = G.subgraph((u['url_name'] for u in user_info if u['iq'] > 0))
    return user_info, G

def load_artists():
    with open(f"{DATAPATH}/artist_info.jl", 'r') as f:
        artist_info = [json.loads(line) for line in f.readlines()]
        for a in artist_info:
            del a['songs']
    return artist_info

def load_song_info():
    with open(f'{DATAPATH}/song_info.jl', 'r') as f:
        song_info = f.readlines()
        song_info = list(map(json.loads, song_info))
    return song_info

def load_annotation_info(reviewed=True):
    '''
        If reviewed, then only return reviewed annotations.
    '''
    annotation_info = []
    with open(f"{DATAPATH}/annotation_info.jl", 'r') as f:
        for line in f:
            j = json.loads(line)
            if j['type'] == 'reviewed' or not reviewed:
                annotation_info.append(j)
    return annotation_info

def load_lyrics_info():
    lyrics_info = []
    with open(f'{DATAPATH}/lyrics.jl', 'r') as f:
        for line in f:
            lyrics_info.append(json.loads(line))
    return lyrics_info
