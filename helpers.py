import bisect
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup
import collections

def make_cont_bins(x, y, num_bins=100, log=False, lower=-10):
    """
    lower is start of logpsace if x takes nonpositive values
    """
    min_x, max_x = min(x), max(x)
    if not log:
        bin_range = np.linspace(min_x, max_x, num_bins)
    else:
        if min_x > 0:
            bin_range = np.logspace(np.log10(min_x), np.log10(max_x), num_bins)
        else:
            bin_range = np.logspace(lower, np.log10(max_x), num_bins)
    bins = [[] for _ in range(num_bins)]
    for i in range(len(x)):
        ind = bisect.bisect_left(bin_range, x[i])
        ind = min(ind, len(bins)-1)
        bins[ind].append(y[i])
    return bin_range, bins

def make_bins(x, y, bin_size=1):
    """
    For use when x takes not too many values
    (e.g. a few hundred ints, maybe a few thousand)
    """
    ran = range((max(x)+min(x)+1))
    unit_bins = [[] for _ in ran]
    for j in range(len(x)):
        unit_bins[x[j]].append(y[j])
    bin_range = range(0, max(x)+1-bin_size, bin_size)
    bins = [[] for _ in bin_range]
    for k, lst in enumerate(bins):
        for l in range(bin_size):
            lst.extend(unit_bins[k*bin_size + l])
    return bin_range, bins

def to_dt_s(date):
    """ song release date to datetime """
    return datetime.strptime(date, '%b %d, %y')   

def to_dt_a(date):
    """ annotation time to datetime object """
    return datetime.strptime(date, '%b %d, %Y %I:%M:%S %p') 

"""measures of quality, e.g.
quality = num_quality_tags(content, quality_tags)
quality = len(BeautifulSoup(content).get_text())
where content is annotation content
"""
quality_tags = ['a', 'img', 'iframe', 'blockquote', 'twitter-widget',
                'ul','ol', 'embedly-embed']
tags = set(['html', 'body', 'br', 'div', 'p', 'em', 'h3',
           'embedly-youtube', 'strong', 'hr', 
           'small', 'li', 'h2', 'sup', 'i', 'b',
           'center', 'span'] + quality_tags)

def num_quality_tags(content, quality_tags):
    soup = BeautifulSoup(content, features='lxml')
    return len([t.name for t in soup.find_all() if t.name in quality_tags])

def array_to_latex(A):
    """ Converts numpy array into latex formatted tabular

    From https://tex.stackexchange.com/questions/54990/convert-numpy-array-into-tabular
    """
    print(" \\\\\n".join([" & ".join(map(str,line)) for line in A]))
    
def make_user_to_ann_idx(annotation_info):
    """
    annotations sorted by timestamp
    """
    user_to_ann_idx = collections.defaultdict(list)
    for idx, a in enumerate(annotation_info):
        if a['type'] == 'reviewed':
            u = a['edits_lst'][-1]['name'].split('/')[-1]
            user_to_ann_idx[u].append(idx)
    for u in user_to_ann_idx:
        user_to_ann_idx[u].sort(key=lambda idx:
                               to_dt_a(
                                annotation_info[idx]['time']
                               ).timestamp())
    return user_to_ann_idx

def make_song_to_ann_idx(annotation_info):
    """ 
    annotations sorted by timestamp
    """
    song_to_ann_idx = collections.defaultdict(list)
    for idx, a in enumerate(annotation_info):
        if a['type'] == 'reviewed':
            s = a['song']
            song_to_ann_idx[s].append(idx)
    for s in song_to_ann_idx:
        song_to_ann_idx[s].sort(key=lambda idx:
                               to_dt_a(
                                annotation_info[idx]['time']
                               ).timestamp())
    return song_to_ann_idx
