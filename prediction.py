import collections
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import scipy.stats as stats

from load_data import *
from helpers import *
from constants import *


def qtile_labels(users, bot_qtile, top_qtile):
    """ Splits input users into two groups by iq.
    
    Returns lab_users, which is a list of users that
    are labeled, and an array Y of size len(lab_users),
    with Y[k]=1 if user is at least top_qtile IQ quantile,
    and Y[k]=0 if user is at most bot_qtile IQ quantile.
    """
    Y = []
    user_to_iq = {u['url_name']:u['iq'] for u in user_info}
    user_iqs = [user_to_iq[u] for u in users]
    top_bound = np.quantile(user_iqs, top_qtile)
    bot_bound = np.quantile(user_iqs, bot_qtile)
    for iq in user_iqs:
        if iq >= top_bound:
            Y.append(1)
        elif iq < bot_bound:
            Y.append(0)
        else:
            Y.append(-1)
            
    Y = np.array(Y)
    lab_users = np.array(users)[np.where(Y!=-1)]
    Y = Y[Y!=-1]
    return Y, lab_users

def make_top_users(min_annots=30, min_edits=30, 
                   bot_qtile=.3333, top_qtile=.6666, print_stats=True):
    """ Splits users into two groups based on IQ
    
    Only considers users with at least min_annots annotations and
    min_edits edits. Calls qtile_labels for splitting on IQ.
    If print_stats, prints basic statistics about labeled users.
    """
    ann_count = collections.Counter()
    edit_counts = collections.Counter()
    user_to_iq = {u['url_name']:u['iq'] for u in user_info}
    for a in annotation_info:
        if a['type'] == 'reviewed':
            user = a['edits_lst'][-1]['name'].split('/')[-1]
            ann_count[user] += 1
            for e in a['edits_lst'][:-1]:
                user = e['name'].split('/')[-1]
                edit_counts[user] += 1
    
    top_users = []
    for user, count1 in ann_count.items():
        count2 = edit_counts[user]
        if count1 >= min_annots and count2 >= min_edits:
            top_users.append(user)
    top_users = [u for u in top_users if u in user_to_iq]
    Y, lab_users = qtile_labels(top_users, bot_qtile, top_qtile)
    if print_stats:
        print('Number of labeled users:', len(lab_users))
        print('Percentile iq of min:', stats.percentileofscore(list(user_to_iq.values()),
                                      np.min([user_to_iq[u] for u in top_users])))
        print('Total percentile iq of 33 percentile:', stats.percentileofscore(list(user_to_iq.values()),
                                      np.percentile([user_to_iq[u] for u in top_users], 1/3*100)))
        print('Total percentile iq of 66 percentile:', stats.percentileofscore(list(user_to_iq.values()),
                                      np.percentile([user_to_iq[u] for u in top_users], 2/3*100)))
        print('Mean top user annots:', np.mean([ann_count[u] for u in top_users]))
        print('Mean top user edits:', np.mean([edit_counts[u] for u in top_users]))
        
    return top_users, Y, lab_users

def data_matrix(lab_users, max_annots=15, max_a_edits=15, song_to_score=None):
    """ Builds matrix of user data for prediction.
    
    lab_users is filtered so only labeled ones left
    song_to_score is a dictionary mapping song names
    to song novelty scores.
    """
    n = len(lab_users)
    X = [[] for _ in range(n)]
    # user to annotation indices
    user_to_ann_idx = collections.defaultdict(list)
    user_to_edits = collections.defaultdict(list)
    song_to_ann_idx = collections.defaultdict(list)
    ann_idx_to_time_rank = {}
    for i, a in enumerate(annotation_info):
        if a['type'] == 'reviewed':
            song_name = a['song']
            song_to_ann_idx[song_name].append(i)
            u = a['edits_lst'][-1]['name'].split('/')[-1]
            if u in lab_users:
                user_to_ann_idx[u].append(i)
            for time_rank, e in enumerate(list(reversed(a['edits_lst']))[1:]):
                time_rank = time_rank+2
                if u in lab_users:
                    user_to_edits[u].append((i, time_rank))
                
    # compute time ranks
    for _, idx_lst in song_to_ann_idx.items():
        sorted_idx = sorted(idx_lst, 
                            key=lambda i: 
                            to_dt_a(annotation_info[i]['time']).timestamp())
        for time_rank, idx in enumerate(sorted_idx):
            ann_idx_to_time_rank[idx] = time_rank+1
            
    # compute features
    
    for i, u in enumerate(lab_users):
        # annotation features
        ann_quality_tags = []
        time_between_annots = []
        first_ann = []
        song_scores = []
        
        first_annot_idxs = sorted(user_to_ann_idx[u], 
               key=lambda idx: to_dt_a(annotation_info[idx]['time']).timestamp()
              )[:max_annots]
        for j, idx in enumerate(first_annot_idxs):
            time_rank = ann_idx_to_time_rank[idx]
            a = annotation_info[idx]
            content = a['edits_lst'][-1]['content']
            ann_quality_tags.append(num_quality_tags(content, quality_tags))
            if j == 0:
                prev_annot_time = to_dt_a(a['time']).timestamp()
                continue
            curr_annot_time = to_dt_a(a['time']).timestamp()
            time_between_annots.append(curr_annot_time - prev_annot_time)
            prev_annot_time = curr_annot_time
            first_ann.append(int(time_rank==1))
            num_ann_on_song = len(song_to_ann_idx[a['song']])
            prop_time = time_rank/(num_ann_on_song-1) if num_ann_on_song > 1 else 0
            if song_to_score and a['song'] in song_to_score:
                song_scores.append(song_to_score[a['song']])
        
        X[i].append(np.mean(ann_quality_tags))
        X[i].append(np.mean(time_between_annots))
        X[i].append(np.mean(first_ann))
        if song_to_score:
            X[i].append(np.mean(song_scores))
        
        # edit features
        first_edits = sorted(user_to_edits[u],
                                key=lambda t:
                                to_dt_a(
                                    list(reversed(
                                        annotation_info[t[0]]['edits_lst']))[t[1]-1]['time']).timestamp()
                            )[:max_a_edits]
        time_between_edits = []
        first_edit_prop = []
        for j, (idx, time_rank) in enumerate(first_edits):
            a = annotation_info[idx]
            e = list(reversed(a['edits_lst']))[time_rank-1]
            prev_edit = list(reversed(a['edits_lst']))[time_rank-2]
            prev_time = to_dt_a(prev_edit['time']).timestamp()
            curr_time = to_dt_a(e['time']).timestamp()
            prev_content = prev_edit['content']
            prev_ann_len = len(BeautifulSoup(prev_content, features='lxml').get_text())
            ann_len = len(BeautifulSoup(e['content'], features='lxml').get_text())
            prop_time = (time_rank-1)/(len(a['edits_lst'])-1)
            first_edit_prop.append(int(time_rank==2))
            # prev difference features
            if j == 0:
                prev_edit_time = to_dt_a(e['time']).timestamp()
                continue
            curr_edit_time = to_dt_a(e['time']).timestamp()
            time_between_edits.append(curr_edit_time - prev_edit_time)
            prev_edit_time = curr_edit_time
            prev_edit_user = list(reversed(a['edits_lst']))[time_rank-2]['name'].split('/')[-1]
            
        X[i].append(np.mean(time_between_edits))
        X[i].append(np.mean(first_edit_prop))
        
    return np.array(X)

def bootstrapped_logistic(X, Y, num_samples=10000, alpha=.05):
    """ Bootstrapped logistic model for super vs. normal experts
    
    Prints coefficients means, standard deviations,
    and endpoints of (1-alpha) confidence intervals.
    """
    coeffs = []
    n = X.shape[0]
    for _ in range(num_samples):
        inds = np.random.choice(range(n), n, replace=True)
        Xboot = X[inds,:]
        Yboot = Y[inds]
        clf = LogisticRegression(penalty='none', solver='lbfgs')
        clf.fit(Xboot, Yboot)
        coeffs.append(np.hstack((
                    clf.intercept_, np.squeeze(clf.coef_))))
    coeffs = np.array(coeffs)
    print('Means:', np.mean(coeffs, axis=0))
    print('STDev:', np.std(coeffs, axis=0))
    print('Lower:', np.quantile(coeffs, alpha/2, axis=0))
    print('Upper:', np.quantile(coeffs, 1-alpha/2, axis=0))


def clf_results(X, Y, clf, num_trials=1000, prop_train=.75):
    accuracys, aucs = [], []
    baselines = []
    for _ in range(num_trials):
        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, train_size=prop_train)
        Xtr_scaled = preprocessing.scale(Xtr)
        Xte_scaled = preprocessing.scale(Xte)
        clf.fit(Xtr_scaled, Ytr)
        preds = clf.predict(Xte_scaled)
        accuracy = clf.score(Xte_scaled, Yte)
        probs = clf.predict_proba(Xte_scaled)
        auc = roc_auc_score(Yte, probs[:,1])
        accuracys.append(accuracy)
        aucs.append(auc)
        # majority-class accuracy baseline
        baselines.append(max(np.count_nonzero(Yte==0)/Yte.shape[0], 
                             np.count_nonzero(Yte==1)/Yte.shape[0]))
    print('Accuracy:', np.mean(accuracys),
          '| STDev:', np.std(accuracys))
    print('AUC:', np.mean(aucs),
          '| STDev:', np.std(aucs))
    print('Baseline accuracy:', np.mean(baselines))
    

if __name__ == '__main__':
    # Load data
    print('~~~Loading data~~~')
    user_info, _ = load_graph()
    annotation_info = load_annotation_info()

    # Label users
    print('~~~Labeling users~~~')
    min_annots, min_edits = 30, 30
    top_users, Y, lab_users = make_top_users(min_annots, min_edits)

    # Make data matrix
    print('~~~Making data matrix~~~')
    max_used_annots = 15
    max_used_edits = 15
    X = data_matrix(lab_users, max_used_annots, max_used_edits)

    # Bootstrapped logistic model
    print('~~~Bootstrapped logistic model~~~')
    bootstrapped_logistic(preprocessing.scale(X), Y, num_samples=10000)

    # Prediction
    print('~~~Prediction~~~')
    num_trials = 1000
    clf = LogisticRegression(penalty='none', solver='lbfgs')
    for k in range(X.shape[1]):
        print(f'~~~{k+1} features~~~')
        clf_results(X[:,:k+1], Y, clf, num_trials=num_trials)

