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
        plt.xlabel('q')
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
        count += 1
        namestr = f'{FIGPATH}/figure5-0-{count}.pdf'
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
            plt.xlabel('R')
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

    valid_figures = (2,3,5,8,10)
    print(f'Valid figures are: {valid_figures}')
    if args.figure in valid_figures:
        print('~~~Loading data~~~')
        annotation_info = load_annotation_info()
        print(f'Computing Figure {args.figure}')
    if args.figure == 2:
        figure2()
    elif args.figure == 3:
        figure3()
    elif args.figure == 5:
        figure5a()
    elif args.figure == 8:
        figure8a()
    elif args.figure == 10:
        figure10()
    else:
        print('Invalid figure number')



