import random
import time
import json
import csv
import networkx as nx
import numpy as np
import scipy
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import collections
from datetime import datetime
import re
from bs4 import BeautifulSoup
import argparse

# our modules
from helpers import *
from load_data import *
from constants import *
from process_lyrics import *
from plot_settings import one_col, two_col

def figure2():
    def tot_user_annots_distr():
        user_to_anns = collections.defaultdict(list)
        for i, a in enumerate(annotation_info):
                if a['type'] == 'reviewed':
                    u = a['edits_lst'][-1]['name']
                    user_to_anns[u].append(i)
        data = []
        for anns in user_to_anns.values():
            data.append(len(anns))
        x, y = np.unique(data, return_counts=True)
        plt.plot(x, y, 'o', color='lightseagreen')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Annotations')
        plt.ylabel('Users')
        
    def tot_annots_distr():
        song_to_anns = collections.defaultdict(list)
        for i, a in enumerate(annotation_info):
                if a['type'] == 'reviewed':
                    song_name = a['song']
                    song_to_anns[song_name].append(i)
        data = []
        for anns in song_to_anns.values():
            data.append(len(anns))
        x, y = np.unique(data, return_counts=True)
        plt.plot(x, y, 'o', color='lightseagreen')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Annotations')
        plt.ylabel('Songs')

    two_col()

    plt.figure()
    tot_user_annots_distr()
    plt.tight_layout(pad=0.2)
    plt.xticks([1, 10, 100, 1000])
    plt.yticks([1, 10, 100, 10**3, 10**4])
    plt.minorticks_off()
    namestr = f'{FIGPATH}/figure2-0.pdf'
    plt.savefig(namestr)
    print(f'Saved to {namestr}')

    plt.figure()
    tot_annots_distr()
    plt.tight_layout(pad=0.2)
    plt.xticks([1, 10, 100])
    plt.yticks([1, 10, 100, 10**3])
    plt.minorticks_off()
    namestr = f'{FIGPATH}/figure2-1.pdf'
    plt.savefig(namestr)
    print(f'Saved to {namestr}')

def figure3():
    def plot_prop_covered_vs_stat(stat, song_to_lyrics, song_to_ann_lyrics):
        song_to_stat = {s['url_name']:s[stat] for s in song_info_gen()}
        x, y = [], []
        for s in song_to_lyrics:
            if s in song_to_stat:
                covered_val = covered_stat(s, song_to_lyrics, song_to_ann_lyrics)
                xval = song_to_stat[s]
                if xval is not None:
                    x.append(xval)
                    y.append(covered_val)
        
        x, y = make_cont_bins(x, y, num_bins=250, log=True)
        y = [np.mean(b) for b in y]
        fig, ax = plt.subplots()
        ax.plot(x,y, 'o', color='lightseagreen')
        plt.xscale('log')
        plt.xlabel(stat.title())
        plt.ylabel('coverage'.title())
        if stat == 'contributors':
            plt.xlim(.5, 3000)
            ax.set_xticks([1, 10, 10**2, 10**3])
        elif stat == 'views':
            plt.xlim(3000, max(x)*1.5)
            ax.set_xticks([10**4, 10**5, 10**6, 10**7])
        plt.ylim(0,1.05)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    two_col()
    song_to_lyrics, song_to_ann_lyrics = make_song_to_lyrics(annotation_info)
    stat = 'contributors'
    plot_prop_covered_vs_stat(stat, song_to_lyrics, song_to_ann_lyrics)
    plt.tight_layout(pad=0)
    namestr = f'{FIGPATH}/figure3-0.pdf'
    plt.savefig(namestr)
    print(f'Saved to {namestr}')

    stat = 'views'
    plot_prop_covered_vs_stat(stat, song_to_lyrics, song_to_ann_lyrics)
    plt.tight_layout(pad=0)
    namestr = f'{FIGPATH}/figure3-1.pdf'
    plt.savefig(namestr)
    print(f'Saved to {namestr}')

def figure4():
    def text_coverage_over_views(ax, ts, end_ct=5):
        """song coverage against views, split by lyric originality
        """
        song_to_lyrics, song_to_ann_lyrics = make_song_to_lyrics(annotation_info)
        song_to_score = compute_song_to_score(song_to_lyrics)
        song_to_views = {s['url_name']:s['views'] for s in song_info_gen()}
        ret = collections.defaultdict(list)
        stypes = ('high', 'low')
        highpercent = 2/3
        lowpercent = 1/3
        high_cut = np.quantile(list(song_to_score.values()),highpercent)
        low_cut = np.quantile(list(song_to_score.values()),lowpercent)
        song_to_stype = {}
        ctr = collections.Counter()
        curr_highers = collections.defaultdict(set)
        k = 0
        for i, s in enumerate(song_to_lyrics):
            coverage = covered_stat(s, song_to_lyrics, song_to_ann_lyrics)
            views = song_to_views.get(s,0)
            if views:
                score = song_to_score[s]
                if score >= high_cut:
                    stype = stypes[0]
                    ctr[0] += 1
                elif score <= low_cut:
                    stype = stypes[1]
                    ctr[1] += 1
                else:
                    continue
                song_to_stype[s] = stype
                curr_highers[stype].add((views, coverage, k))
                k += 1
        for i, t in enumerate(ts[1:]):
            for stype in stypes:
                tot = len(curr_highers[stype])
                if tot >= end_ct:
                    coverages = [trip[1] for trip in curr_highers[stype]]
                    tot_coverages = sum(coverages)
                    ret[stype].append(tot_coverages/tot) # mean
                    curr_highers[stype] = [trip for trip in curr_highers[stype] if trip[0] >= t]
        for stype in stypes:
            data = ret[stype]
            plt.plot(ts[:len(data)], data, linewidth=1.5)
        plt.legend(['high originality', 'low originality'], fontsize='small', 
                   loc='upper left', fancybox=False, framealpha=1,
                      edgecolor='k', facecolor='oldlace')
        
    one_col()
    lower = 5000
    upper = 5*10**6
    ts = range(lower, upper, 5000)
    fig, ax = plt.subplots(1,figsize=(2,1.3))
    ax.set_prop_cycle('color', CYCLE_COLORS)
    text_coverage_over_views(ax, ts, end_ct=10)
    plt.xscale('log')
    plt.xlim(lower, upper)
    plt.ylim(.25, .82)
    plt.xlabel('Minimum views')
    plt.ylabel('Coverage')
    plt.yticks([.3, .4, .5, .6, .7, .8])
    plt.grid()
    plt.tight_layout(pad=0)
    plt.minorticks_off()
    plt.savefig(f'{FIGPATH}/figure4.pdf')
        

def figure5a():
    def prop_ann_time_rank_to_user_stat(stat, min_annots=10, max_annots=100):
        song_to_annot_idx = collections.defaultdict(list)
        x, y = [], []
        if stat in STATS:
            user_to_stat = {u['url_name']:u['stats'][stat] for u in user_info_gen()}
        else:
            user_to_stat = {u['url_name']:u[stat] for u in user_info_gen()}
        for i, a in enumerate(annotation_info):
            if a['type'] == 'reviewed':
                song_name = a['song']
                song_to_annot_idx[song_name].append(i)
        for s, idx_lst in song_to_annot_idx.items():
            if min_annots <= len(idx_lst) <= max_annots:
                sorted_idx = sorted(idx_lst, 
                                    key=lambda i: 
                                    to_dt_a(annotation_info[i]['time']).timestamp())
                for time_rank, i in enumerate(sorted_idx):
                    u = annotation_info[i]['edits_lst'][-1]['name']
                    u = u.split('/')[-1]
                    val = user_to_stat.get(u, 0)
                    x.append(time_rank/(len(idx_lst)-1))
                    y.append(val)
        x, y = make_cont_bins(x, y, num_bins=19)
        y = [np.mean(b) for b in y]
        plt.plot(x, y, '-o', color=PRED)
        plt.xlabel('Prop. time rank (q)')
        plt.ylabel(stat)
        plt.xticks([0, .5, 1])
        if stat == 'iq':
            plt.ylabel('IQ')
            plt.yticks([50000, 60000, 70000], 
                       ['50k', '60k', '70k'])
    two_col()
    count = 0
    for stat in PUB_STATS:
        plt.figure()
        prop_ann_time_rank_to_user_stat(stat, min_annots=10, max_annots=100)
        plt.tight_layout(pad=0)
        namestr = f'{FIGPATH}/figure5-0-{count}.pdf'
        count += 1
        plt.savefig(namestr)
        print(f'Saved to {namestr}')

def figure5b():
    def prop_annotation_time_rank_to_quality(ptype, min_annots=10, max_annots=100):
        song_to_annot_idx = collections.defaultdict(list)
        x, y = [], []
        for i, a in enumerate(annotation_info):
            if a['type'] == 'reviewed':
                song_name = a['song']
                song_to_annot_idx[song_name].append(i)
        for s, idx_lst in song_to_annot_idx.items():
            if min_annots <= len(idx_lst) <= max_annots:
                sorted_idx = sorted(idx_lst, 
                                    key=lambda i: 
                                    to_dt_a(annotation_info[i]['time']).timestamp())
                for time_rank, i in enumerate(sorted_idx):
                    a = annotation_info[i]
                    annot = a['edits_lst'][-1]
                    content = annot['content']
                    if ptype == 'Quality Tags':
                        quality = num_quality_tags(content, quality_tags)
                    elif ptype == 'Length':
                        quality = len(BeautifulSoup(content).get_text())
                    x.append(time_rank/(len(idx_lst)-1))
                    y.append(quality)
       
        x, y = make_cont_bins(x, y, num_bins=15)
        y = [np.mean(b) for b in y]
        plt.plot(x, y, '-o', color=PRED)
        plt.xlabel('Prop. time rank (q)')
        plt.ylabel(ptype)
    two_col()
    count = 0
    for ptype in ('Quality Tags', 'Length'):
        plt.figure()
        prop_annotation_time_rank_to_quality(ptype, min_annots=10, max_annots=100)
        plt.tight_layout(pad=0)
        namestr = f'{FIGPATH}/figure5-1-{count}.pdf'
        plt.savefig(namestr)
        print(f'Saved to {namestr}')
        count += 1

def figure6():
    def ann_prop_time_rank_to_lyric_score(min_annots=10):
        song_to_lyrics, _ = make_song_to_lyrics(annotation_info)
        vectorizer = lyric_vectorizer(song_to_lyrics)
        song_to_annot_idx = collections.defaultdict(list)
        x, y = [], []
        for i, a in enumerate(annotation_info):
            if a['type'] == 'reviewed':
                song_name = a['song']
                song_to_annot_idx[song_name].append(i)
        for s in song_to_annot_idx:
            song_to_annot_idx[s].sort(key=lambda i:
                                to_dt_a(annotation_info[i]['time']).timestamp())
            idx_lst = song_to_annot_idx[s]
            if len(idx_lst) >= min_annots:
                content_lst = [BeautifulSoup(
                    annotation_info[i]['edits_lst'][-1]['content'], 'lxml').get_text()
                              for i in idx_lst ]
                score_matrix = vectorizer.transform(content_lst)
                for tr, i in enumerate(idx_lst):
                    prop_time_rank = tr/(len(idx_lst)-1)
                    if score_matrix[tr, :].nnz:
                        _, _, v = scipy.sparse.find(score_matrix[tr,:])
                        score = (np.percentile(v, 60) + np.percentile(v, 75) + np.percentile(v, 90))/3
                    else:
                        score = 0
                    if score:
                        x.append(prop_time_rank)
                        y.append(score-1) # -1 to align with paper defn
        return x, y
    x, y = ann_prop_time_rank_to_lyric_score(min_annots=10)
    one_col()
    plt.figure(figsize=(2, 1.2))
    xb, yb = make_cont_bins(x, y, num_bins=15)
    yb = [np.mean(b) for b in yb]
    plt.plot(xb, yb, '-o', color=PRED)
    plt.xlabel('Prop. time rank (q)')
    plt.ylabel('Originality')
    plt.yticks([4.9, 4.95, 5, 5.05, 5.1])
    plt.tight_layout(pad=0)
    namestr = 'figures/figure6.pdf'
    plt.savefig(namestr)
    print(f'Saved to {namestr}')

def figure8a():
    def split_editnum_vs_stat(stat, most_edits=10):
        """edit time rank vs stat
        split based on number of edits per annotation
        """
        two_col()
        if stat in STATS:
            user_to_stat = {u['url_name']:u['stats'][stat] for u in user_info_gen()}
        else:
            user_to_stat = {u['url_name']:u[stat] for u in user_info_gen()}
        for edit_num in range(1, most_edits+1):
            ran = range(edit_num)
            edit_bins = [[] for _ in ran]
            for a in annotation_info:
                if a['type'] == 'reviewed' and len(a['edits_lst']) == edit_num:
                    og_user = a['edits_lst'][0]['name'].split('/')[-1]
                    for i, edit in enumerate(reversed(a['edits_lst'])):
                        user = edit['name'].split('/')[-1]
                        if user in user_to_stat:
                            val = user_to_stat[user]
                            edit_bins[i].append(val)
            y = [np.mean(b) for b in edit_bins]
            plt.plot(np.array(ran)+1, y, '-o', 
                      markeredgecolor='k')
            plt.xlabel('Time rank (R)')
            plt.ylabel(stat)
            if stat == 'iq':
                plt.ylabel('IQ')
                plt.yticks([100000, 150000, 200000],
                            ['100k', '150k', '200k'])
            elif stat == 'Annotations':
                pass
                plt.yticks([1000, 1500, 2000],
                           ['1k', '1.5k', '2k'])
                
        plt.xticks(range(1,most_edits+1))
    two_col()
    for count, stat in enumerate(PUB_STATS):
        plt.figure()
        split_editnum_vs_stat(stat, most_edits=10)
        plt.tight_layout(pad=0)
        namestr = f"{FIGPATH}/figure8-0-{count}.pdf"
        plt.savefig(namestr)
        print(f'Saved to {namestr}')

def figure8b():
    def editnum_vs_quality(ptype, most_edits=10):
        edit_bins = [[[] for _ in range(k)] for k in range(1,most_edits+1)]
        data = []
        for a in annotation_info:
            if a['type'] == 'reviewed':
                edits_lst = a['edits_lst']
                edit_num = len(edits_lst)
                if edit_num <= most_edits:
                    for i, edit in enumerate(
                        list(reversed(edits_lst))[:edit_num]):
                        content = edit['content']
                        if ptype == 'Quality Tags':
                            quality = num_quality_tags(content, quality_tags)
                        if ptype == 'Length':
                            quality = len(BeautifulSoup(content).get_text())
                        edit_bins[edit_num-1][i].append(quality)
        for edit_num in range(most_edits):
            y = [np.mean(b) for b in edit_bins[edit_num]]
            plt.plot(range(1, edit_num+2), y, 'o-')
        plt.xlabel('Time rank (R)')
        plt.ylabel(ptype)
        plt.xticks(range(1, most_edits+1))
    two_col()
    for count, ptype in enumerate(('Quality Tags', 'Length')):
        plt.figure()
        editnum_vs_quality(ptype, most_edits=10)
        plt.tight_layout(pad=0)
        namestr = f'{FIGPATH}/figure8-1-{count}.pdf'
        plt.savefig(namestr)
        print(f'Saved to {namestr}')


def figure9a():
    def date_split_user_ann_evolution(ax, min_user_anns=10, valtype=0, max_days = 750):
        ylab = {
        1: 'First ann.',
        4: 'Quality Tags',
        5: 'Length',
        7: 'Lyric Originality',
        8: 'Song Originality'
        }
        user_to_ann_idx = make_user_to_ann_idx(annotation_info)
        song_to_ann_idx = make_song_to_ann_idx(annotation_info)
        ann_idx_to_prop_time = {}
        user_to_iq = {u['url_name']:u['iq'] for u in user_info_gen()}
        xs, ys = collections.defaultdict(list), collections.defaultdict(list)
        for s in song_to_ann_idx:
            num_anns = len(song_to_ann_idx[s])
            for time_rank, idx in enumerate(song_to_ann_idx[s]):
                ann_idx_to_prop_time[idx] = time_rank/(num_anns-1) if num_anns > 1 else 0
        # first annotation datetimes
        user_first_dt = {}
        for u in user_to_ann_idx:
            first_idx = user_to_ann_idx[u][0]
            user_first_dt[u] = to_dt_a(annotation_info[first_idx]['time'])
            
        utypes = ['high iq', 'mid iq', 'low iq', 'all']
        ann_high_tile = 10**5
        ann_midup_tile = 5*10**4
        ann_midlow_tile = 10**4

        for u in user_to_ann_idx:
            num_user_anns = len(user_to_ann_idx[u])
            if num_user_anns >= min_user_anns and u in user_to_iq:
                iq = user_to_iq[u]
                if iq > ann_high_tile:
                    utype = utypes[0]
                elif ann_midup_tile > iq >= ann_midlow_tile:
                    utype = utypes[1]
                else:
                    utype = utypes[2]
                for idx in user_to_ann_idx[u]:
                    a = annotation_info[idx]
                    elapsed = to_dt_a(a['time']) - user_first_dt[u]
                    if elapsed.days > max_days:
                        break
                    content = a['edits_lst'][-1]['content']
                    if valtype==1:
                        val = int(ann_idx_to_prop_time[idx] == 0)
                    elif valtype==4:
                        val = num_quality_tags(content, quality_tags)
                    elif valtype==5:
                        val = len(BeautifulSoup(content).get_text())
                    elif valtype==7:
                        val = ann_idx_to_score[idx]
                    elif valtype==8:
                        val = song_to_score.get(a['song'], None)
                    if val is not None:
                        xs[utype].append(elapsed.days)
                        ys[utype].append(val)
                        xs['all'].append(elapsed.days)
                        ys['all'].append(val)
        for utype in utypes:
            if utype == 'low iq':
                continue
            x, y = xs[utype], ys[utype]
            x, y = make_cont_bins(x, y, num_bins=15)
            # cumulative average
            yout = [[] for _ in range(len(y))]
            for i in range(len(y)):
                for j in range(i+1):
                    yout[i].extend(y[j])
                yout[i] = np.mean(yout[i])
            # omits very first annotation
            ax.plot(x[1:], yout[1:], '-o')
        plt.xlabel('User Age')
        plt.ylabel(ylab[valtype])
        plt.xticks([0,250,500,750])
        if valtype == 1:
            plt.yticks([.08, .1, .12], ['.08', '.1', '.12'])

    song_to_lyrics, song_to_ann_lyrics = make_song_to_lyrics(annotation_info)
    song_to_score = compute_song_to_score(song_to_lyrics)
    ann_idx_to_score = compute_ann_idx_to_score(song_to_lyrics, annotation_info)
    two_col()
    for valtype in (1,4,5,7,8):
        fig, ax = plt.subplots(1)
        ax.set_prop_cycle('color', CYCLE_COLORS)
        date_split_user_ann_evolution(ax, min_user_anns=10, valtype=valtype)
        plt.tight_layout(pad=0)
        namestr = f'{FIGPATH}/figure9-0-{valtype}.pdf'
        plt.savefig(namestr)
        print(f'Saved to {namestr}')

def figure9b():
    def date_user_edit_evolution(min_user_edits=10, valtype=0, max_days=750):
        ylab = {
        0: 'First edits',
        }
        user_to_edits = collections.defaultdict(list)
        user_to_iq = {u['url_name']:u['iq'] for u in user_info_gen()}
        utypes = ['high iq', 'mid iq', 'low iq', 'all']
        for i, a in enumerate(annotation_info):
            if a['type'] == 'reviewed':
                song_name = a['song']
                u = a['edits_lst'][-1]['name'].split('/')[-1]
                for time_rank, e in enumerate(list(reversed(a['edits_lst']))[1:]):
                    true_time_rank = time_rank+2
                    user_to_edits[u].append((i, true_time_rank))
                    
        # first edit datetimes
        user_first_dt = {}
        for u in user_to_edits:
            user_to_edits[u].sort(key = lambda t:
                                 to_dt_a(
                                     annotation_info[t[0]]['edits_lst'][::-1][t[1]-1]['time']).timestamp()
                                 )
            first_idx, true_time_rank = user_to_edits[u][0]
            user_first_dt[u] = to_dt_a(
                list(reversed(annotation_info[first_idx]['edits_lst']))[true_time_rank-1]['time']
                    )
        
        xs, ys = collections.defaultdict(list), collections.defaultdict(list)
        for u in user_to_edits:
            num_user_edits = len(user_to_edits[u])
            if num_user_edits >= min_user_edits and u in user_to_iq:
                iq = user_to_iq[u]
                if iq > 1e5:
                    utype = utypes[0]
                elif iq > 1e4:
                    utype = utypes[1]
                else:
                    utype = utypes[2]
                for life_time_rank, (idx, true_time_rank) in enumerate(user_to_edits[u]):
                    time_rank = true_time_rank-1
                    a = annotation_info[idx]
                    o_edits_lst = a['edits_lst'][::-1]
                    elapsed = to_dt_a(o_edits_lst[time_rank]['time']) - user_first_dt[u]
                    if elapsed.days > max_days:
                        break
                    
                    if valtype==0:
                        val = int(time_rank == 1)
                    
                    xs[utype].append(elapsed.days)
                    ys[utype].append(val)
                    xs['all'].append(elapsed.days)
                    ys['all'].append(val)
        for utype in utypes:
            if utype == 'low iq':
                continue
            x, y = xs[utype], ys[utype]
            x, y = make_cont_bins(x, y, num_bins=21)
            yout = [[] for _ in range(len(y))]
            for i in range(len(y)):
                for j in range(i+1):
                    yout[i].extend(y[j])
                yout[i] = np.mean(yout[i])
            # omits very first edit
            plt.plot(x[1:], yout[1:], '-o')
        plt.xlabel('User Age')
        plt.ylabel(ylab[valtype])
        plt.xticks([0,250,500,750])
        plt.yticks([.30, .35], ['.30', '.35'])
    valtype = 0
    two_col()
    fig, ax = plt.subplots(1)
    ax.set_prop_cycle('color', CYCLE_COLORS)
    date_user_edit_evolution(min_user_edits=10, valtype=valtype)
    plt.tight_layout(pad=0)
    namestr = f'{FIGPATH}/figure9-1-{valtype}.pdf'
    plt.savefig(namestr)
    print(f'Saved to {namestr}')


def figure10():
    def rank_vs_stat(ax, stat):
        G = load_graph(positive_iq=True)
        r = nx.pagerank(G)
        if stat in STATS:
            name_to_stat = {u['url_name']: u['stats'][stat] for u in user_info_gen()}
        else:
            name_to_stat = {u['url_name']: u[stat] for u in user_info_gen()}
        x, y = [], []
        for v in G.nodes:
            x.append(name_to_stat[v])
            y.append(r[v])

        # use lighter purple
        ax.plot(x, y, 'o', alpha=.1, color='#8F8FEB')
        
        x, y = make_cont_bins(x, y, num_bins=150, log=True)
        y = [np.mean(b) if b else 0 for b in y]
        ax.plot(x, y, 'X', markersize=3, color='#B80717')
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.minorticks_off()
        ax.set_xlim(.8,2*max(x))
        ax.set_ylim(2e-6,3*max(y))
        ax.set_xlabel(stat)
        if stat == 'iq':
            ax.set_xlabel('IQ')
            ax.set_xticks([1, 10**2, 10**4, 10**6])
        elif stat == 'Annotations':
            ax.set_xticks([1, 10**1, 10**2, 10**3, 10**4])
            
    count = 0
    one_col()
    fig, axs = plt.subplots(1, 2, sharey=True)
    axs[0].set_ylabel('PageRank')
    for stat in PUB_STATS:
        axs[count].set_prop_cycle('color', CYCLE_COLORS)
        rank_vs_stat(axs[count], stat)
        count += 1
    plt.tight_layout(pad=.5)
    namestr = f'{FIGPATH}/figure10.png'
    plt.savefig(namestr)
    print(f'Saved to {namestr}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--figure', type=int, default=-1)
    args = parser.parse_args()

    valid_figures = (2,3,4,5,6,8,9,10)
    print(f'Valid figures are: {valid_figures}')
    if args.figure in valid_figures:
        print('~~~Loading data~~~')
        annotation_info = load_annotation_info()
        print(f'Computing Figure {args.figure}')
    if args.figure == 2:
        figure2()
    elif args.figure == 3:
        figure3()
    elif args.figure == 4:
        figure4()
    elif args.figure == 5:
        figure5a()
        figure5b()
    elif args.figure == 6:
        figure6()
    elif args.figure == 8:
        figure8a()
        figure8b()
    elif args.figure == 9:
        figure9a()
        figure9b()
    elif args.figure == 10:
        figure10()
    else:
        print('Invalid figure number')



