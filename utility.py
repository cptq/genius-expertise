import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from load_data import *
from constants import *
from helpers import *
from plot_settings import two_col

def make_user_to_prop_times(min_user_anns=10, min_anns=5):
    """ computes dictionaries from user to annotation prop times
    
    min_user_anns is the minimum number of annotations for users
    min_anns is minimum number of annotations for songs
    """
    user_to_ann_idx = make_user_to_ann_idx(annotation_info)
    song_to_ann_idx = make_song_to_ann_idx(annotation_info)
    ann_idx_to_prop_time = {}
    user_to_iq = {u['url_name']:u['iq'] for u in user_info}
    xs, ys = collections.defaultdict(list), collections.defaultdict(list)
    for s in song_to_ann_idx:
        num_anns = len(song_to_ann_idx[s])
        if num_anns > min_anns:
            for time_rank, idx in enumerate(song_to_ann_idx[s]):
                ann_idx_to_prop_time[idx] = time_rank/(num_anns-1) if num_anns > 1 else 0
    utypes = ['high iq', 'mid iq', 'low iq']
    utype_counts = collections.Counter()
    user_to_prop_times = {}
    user_to_utype = {}
    
    # users with at least min_user_anns annotations
    user_above_min = {}
    for u in user_to_ann_idx:
        num_user_anns = len(user_to_ann_idx[u])
        if num_user_anns >= min_user_anns and u in user_to_iq:
            user_above_min[u] = user_to_iq[u]
            
    hp = 2/3
    lp = 1/3
    high_tile = np.quantile(list(user_above_min.values()), hp)
    low_tile = np.quantile(list(user_above_min.values()), lp)
    print('low/high iq cutoffs:', low_tile, high_tile)
    for u in user_to_ann_idx:
        num_user_anns = len(user_to_ann_idx[u])
        if num_user_anns >= min_user_anns and u in user_to_iq:
            iq = user_to_iq[u]
            if iq > high_tile:
                utype = utypes[0]
            elif iq > low_tile:
                utype = utypes[1]
            else:
                utype = utypes[2]
            prop_times = []
            for idx in user_to_ann_idx[u]:
                a = annotation_info[idx]
                if idx not in ann_idx_to_prop_time:
                    continue
                prop_time = ann_idx_to_prop_time[idx]
                prop_times.append(prop_time)
                    
            user_to_prop_times[u] = prop_times
            user_to_utype[u] = utype
    return user_to_prop_times, user_to_utype


def fit_utility_curve(x, y):
    """ Computes utility curve parameters."""
    b = cp.Variable(1)
    a1, a2 = cp.Variable(1), cp.Variable(1)
    c1, c2 = cp.Variable(1), cp.Variable(1)
    u = b - a1*x**2 + a2*x + c1*x**2 - c2*x
    obj = cp.sum_squares(u - y)
    constr = [b >= 0,
              a1 >= 0,
              a2 >= 2*a1,
              c1 >= 0,
              c2 >= 2*c1]
    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve()
    return b.value, a1.value, a2.value, c1.value, c2.value, prob

def utility_model(user_to_prop_times, user_to_utype):
    """ Our utility model. """
    two_col()
    count = 0
    fig, axs = plt.subplots(1,2, figsize=(3.1,1.2), sharey=True)
    axs[0].set_ylabel('Density')
    probs = {}
    for utype in ('high iq', 'low iq'):
        data = []
        for u in user_to_prop_times:
            if user_to_utype[u] == utype:
                data.extend(user_to_prop_times[u])
        # make histogram
        y, x, _ = axs[count].hist(data, bins=7, 
                                  density=True, histtype='step',
                                  color=PRED)
        # histogram bin midpoints
        x = [(x[i]+x[i-1])/2 for i in range(1, len(x))]
        
        x, y = np.array(x), np.array(y)
        res = fit_utility_curve(x, y)
        b, a1, a2, c1, c2, prob = res
        probs[utype] = {'b':b, 'a1':a1,
                        'a2':a2, 'c1':c1,
                        'c2':c2, 'prob':prob}
        # x with 0 and 1
        x_ext = np.hstack(([0],x,[1]))
        u = b - a1*x_ext**2 + a2*x_ext + c1*x_ext**2 - c2*x_ext
        
        
        axs[count].set_xlabel('q')
        axs[count].set_xticks([0,.25,.5,.75,1])
        axs[count].set_ylim(.5,1.5)
        axs[count].set_yticks([.5, .75, 1, 1.25])
        axs[count].plot(x_ext, u, '-k', linewidth=1.5)
        axs[count].plot(x, y, 'oy', markersize=3)
        count += 1
    plt.tight_layout(pad=.5)
    return probs

if __name__ == '__main__':
    # Load data
    print('~~~Load data~~~')
    user_info, _ = load_graph()
    annotation_info = load_annotation_info()

    # Precompute dictionaries from user to annotation prop times
    # and from user to type of user (high or low iq)
    print('~~~Precompute~~~')
    result = make_user_to_prop_times(min_user_anns=10)
    user_to_prop_times, user_to_utype = result

    # Utility model
    print('~~~Fit utility model~~~')
    probs = utility_model(user_to_prop_times, user_to_utype)
    plt.savefig(f'{FIGPATH}/figure7.pdf')
    parameters = ('b', 'a1', 'a2', 'c1', 'c2')
    for utype in ('high iq', 'low iq'):
        print(f'~~~{utype}~~~')
        sol = probs[utype]
        for p in parameters:
            print(p, sol[p])
