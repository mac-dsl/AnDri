import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from io import StringIO
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import pickle
import os
from scipy.stats import gamma
from scipy.optimize import curve_fit
from util.util_exp import result_f1_acc

BASE_URL = "https://dd.weather.gc.ca/today/climate/observations/hourly/csv/ON/"
DAILY_URL ="https://dd.weather.gc.ca/today/climate/observations/daily/csv/ON/"
PROVINCES = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT']

param_types = {
    'l': int,
    'k': int,
    'nm': int,
    'Wmax': int,
    'delta': int,
    'anom': int,
    'rmin': float,
    'd': str,
    'linkage': str
}

DAYS_IN_MONTH = {
    'Jan': 31, 'Feb': 28, 'Mar': 31, 'Apr': 30,
    'May': 31, 'Jun': 30, 'Jul': 31, 'Aug': 31,
    'Sep': 30, 'Oct': 31, 'Nov': 30, 'Dec': 31
}

def _read_station_metadata(path):
    with open(path, encoding="latin1") as f:
        lines = f.readlines()

    ## the header and value columns vary. Should check it before
    meta_header = lines[3]
    meta_value = lines[4]

    meta_df = pd.read_csv(
        StringIO(meta_header + meta_value),
        quotechar='"'
    )

    return meta_df.iloc[0]  # Series

def _read_main_table(path):
    with open(path, encoding="latin1") as f:
        lines = f.readlines()

    table_text = "".join(lines[13:])    ## for safety

    df = pd.read_csv(
        StringIO(table_text),
        quotechar='"'
    )

    return df

def get_normal_data(f):
    meta = _read_station_metadata(f)
    station_name = meta["STATION_NAME"]
    province = meta["PROVINCE"]
    
    df = _read_main_table(f)
    df = df.rename(columns={df.columns[0]: "Variable"})

    month_cols = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec","Year","Code"]

    is_title_row = df[month_cols].isna().all(axis=1)

    keywords = ["Precipitation", "Rainfall", "Snowfall"]

    is_target = df["Variable"].str.contains(
        "|".join(keywords),
        case=False,
        na=False
    )

    precip_df = df[~is_title_row & is_target].copy()
    precip_df["STATION_NAME"] = station_name
    precip_df["PROVINCE"] = province

    return precip_df

def get_links(url):
    r = requests.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    return [a["href"] for a in soup.find_all("a", href=True)]

def list_csv_files(base_url=DAILY_URL, day=True):
    r = requests.get(base_url)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    links = soup.find_all("a")
    csv_files = []
    df = pd.DataFrame(columns=['stationID', 'year'])
    idx = 0


    for a in links:
        href = a.get("href")
        if href and href.endswith(".csv"):
            # full URL
            full_url = base_url + href
            csv_files.append(full_url)
            # print(href)
            if day:
                df.loc[idx] = [href.split('_')[3], int(href.split('_')[4].split('-')[0])]
            else:
                df.loc[idx] = [href.split('_')[3], int(href.split('_')[4])]
            idx +=1
    return csv_files, df


def load_csv_from_url(url, day=False):
    if day:
        df = pd.read_csv(url, parse_dates=['Date/Time'], encoding="cp949")
    else:
        df = pd.read_csv(url, parse_dates=['Date/Time (LST)'], encoding="cp949")
    return df


###############################################################################
## To read exp. results and analyze
def get_andri_param(fname):
    pattern = r'_(' + '|'.join(param_types.keys()) + r')_([a-zA-Z0-9.\-]+)'
    matches = re.findall(pattern, fname)

    params = {}
    for k, v in matches:
        type_func = param_types[k]
        params[k] = type_func(v)
    # print(params)
    return params

def save_pickle(filename, var):   
    with open(filename, 'wb') as f:
        pickle.dump(var, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        var = pickle.load(f)
    return var

## Load scores and model files
def load_score_clf(dir, sel_data, method, conditions, province=None, sel_id=None):
    filelists = os.listdir(f'{dir}{sel_data}/')
    f_lists = [f for f in filelists if f.startswith(method)]
    f_list_score = [f for f in f_lists if f.endswith('scores.pickle')]
    f_list_score.sort()
    # print(f_list_score)
    if 'AnDri' in method:
        f_list_off = [f for f in f_lists if f.endswith('off_.pickle')]
        f_list_off.sort()
        f_list_on = [f for f in f_lists if f.endswith('on_.pickle')]
        f_list_on.sort()
        # print(f_list_on)
        # print(f_list_off)

    elif method in ['NormA', 'SAND']:
        f_list_clf = [f for f in f_lists if f.endswith('clf_.pickle')]
        f_list_clf.sort()

    if sel_data == 'climate':
        stations_s = [f for f in f_list_score if f.split('_')[2]==province]
        sub_list_s = list(set([f for f in stations_s if f.split('_')[3] == str(sel_id)]))

        if 'AnDri' in method:
            print('Conditions:', conditions)
            stations_off = [f for f in f_list_off if f.split('_')[2]==province]
            sub_list_off = list(set([f for f in stations_off if f.split('_')[3] == str(sel_id)]))

            stations_on = [f for f in f_list_on if f.split('_')[2]==province]
            sub_list_on = list(set([f for f in stations_on if f.split('_')[3] == str(sel_id)]))

            # print(len(sub_list_off), len(sub_list_on))

            for sub_s, sub_off, sub_on in zip(sub_list_s, sub_list_off, sub_list_on):
                params = get_andri_param(sub_s)
                # print(params)
                # print(conditions)
                try:
                    if conditions['d'] == params['d'] and conditions['k'] == params['k'] and conditions['linkage'] == params['linkage'] and conditions['Wmax'] == params['Wmax'] and conditions['delta'] == params['delta'] and conditions['rmin'] == params['rmin']:
                        # print(sub_s)
                        clf = []
                        score = load_pickle(f'{dir}{sel_data}/{sub_s}')
                        clf.append(load_pickle(f'{dir}{sel_data}/{sub_off}'))
                        clf.append(load_pickle(f'{dir}{sel_data}/{sub_on}'))
                        break
                except:
                    score, clf  = [], []
        elif method in ['NormA', 'SAND']:
            stations_off = [f for f in f_list_clf if f.split('_')[2]==province]
            sub_list_off = list(set([f for f in stations_off if f.split('_')[3] == str(sel_id)]))
            for sub_s, sub_off in zip(sub_list_s, sub_list_off):
                if re.search(r'_d_([a-zA-Z0-9\-]+)', sub_s).group(1) == conditions['d']:
                    score = load_pickle(f'{dir}{sel_data}/{sub_s}')
                    clf = load_pickle(f'{dir}{sel_data}/{sub_off}')
                    break
        else:
            # stations_off = [f for f in f_list_clf if f.split('_')[2]==province]
            # sub_list_off = list(set([f for f in stations_off if f.split('_')[3] == str(sel_id)]))
            for sub_s in sub_list_s:
                score = load_pickle(f'{dir}{sel_data}/{sub_s}')
                # clf = load_pickle(f'{dir}{sel_data}/{sub_off}')
                clf = []
                break
        # elif 'NormA' in method:
            # stations_off = [f for f in f_list_clf if f.split('_')[2]==province]
            # for sub_s, sub_off in zip(sub_list_s, sub_list_off):
                # if re.search(r'_d_([a-zA-Z0-9\-]+)', sub_s).group(1) == conditions['d']:
                    # score = load_pickle(f'{dir}{sel_data}/{sub_s}')
                    # clf = load_pickle(f'{dir}{sel_data}/{sub_off}')
                    # break


    return score, clf


def gamma_survival(x, k, theta):
    return gamma.sf(x, k, scale=theta)


def fit_gamma_month(df_m, month, min_mm=1.0, k=1.5, theta=5.0):
    thresholds = np.array(df_m.columns, dtype=float)
    exceed_days = df_m.loc[month].values.astype(float)


    mask = thresholds >= min_mm
    t = thresholds[mask]
    exceed_days = exceed_days[mask]


    P_X = exceed_days / DAYS_IN_MONTH[month]
    p = P_X[t== min_mm][0]
    S_Y = P_X / p
    ### Gamma dist. parameters k, theta (init.)
    # p0 = [1.5, 5.0]
    p0 = [k, theta]
    popt, _ = curve_fit(gamma_survival, t, S_Y, p0=p0, bounds=[[0.1, 0.1], [10, 50]])
    k, theta = popt
    return {'month': month, 'k': k, 'theta':theta, 'p_rain':p}


def gamma_monthly_threshold(k, theta, p, alpha=0.02):
    q = 1-alpha/p
    q = np.clip(q, 1e-4, 0.999)
    return gamma.ppf(q, k, scale=theta)


###############################################################################################
def map_station_name(name):
    if name == "LANSDOWNE HOUSE A":
        return "LANSDOWNE HOUSE (AUT)"
    if name == "PEAWANUCK A":
        return "PEAWANUCK (AUT)"
    if name == "ARMSTRONG A":
        return "ARMSTRONG (AUT)"
    if name == "TROIS RIVIERES A":
        return "TROIS-RIVIERES"
    if name == "BAIE-COMEAU":
        return "BAIE-COMEAU A"
    return name

def recompute_score_by_thresholds(method, save_dir, save_to_dir, param, alpha = 0.02):
    # save_dir = './2021_2025_precip_filtered'
    
    filelist_t = os.listdir(save_dir)
    filelist = [f for f in filelist_t if f.endswith('processed.csv')]

    score_dir = '/home/parkj182/research/AnDri_github/results/'
    selected_set = pd.read_csv('./2021_2025_precip_selected/selected_stations.csv', index_col=None)
    # stat_dir = './climate_normal'
    # df_stat = pd.read_csv(f'{stat_dir}/1991-2020_Canadian_Climate_Normals_CANADA_Data.csv')
    # df_name = pd.read_csv(f'{stat_dir}/Canadian_Climate_Normals_1991_2020_station_inventory.csv', index_col=None)

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_map = {
        'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,
        'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12
    }
    if method == 'AnDri':
        method_to = ['AnDri (off)', 'AnDri (on)']
        normalize = param['d']
        k=param['k']
        linkage = param['linkage']
        Wmax = param['Wmax']
        delta = param['delta']
        rmin= param['rmin']

    results = pd.DataFrame()

    for sel_province in PROVINCES:
        sel_filelist = [f for f in filelist if f.split('_')[0] == sel_province]
        # sel_IDs = list(set([f.split('_')[1] for f in sel_filelist]))
        sel_IDs = list(selected_set[selected_set['province']==sel_province]['stationID'])
        sel_IDs.sort()
        print(sel_IDs)

        if method == 'AnDri':
            conditions = {'d': normalize, 'k':k, 'linkage':'ward','Wmax':Wmax, 'delta':delta, 'rmin':rmin}
        elif method in ['NormA', 'SAND']:
            conditions = {'d': param['d']}

        for sel_id_idx in range(len(sel_IDs)):
            print(f'Province: {sel_province}-> {sel_IDs[sel_id_idx]}, Stations: {sel_id_idx+1}/{len(sel_IDs)}')

            ## load scores and models
            score, clf = load_score_clf(score_dir, 'climate', method, conditions, sel_province, sel_IDs[sel_id_idx])

            ## Data
            f_list = [f  for f in filelist if f.split('_')[1] == sel_IDs[sel_id_idx]]
            df_ID = pd.read_csv(f'{save_dir}/{f_list[0]}')

            sel_station = df_ID['Station Name'].iloc[0]
            print(sel_province, sel_IDs[sel_id_idx], sel_station)

            ### compute accuracy with revised labels
            print(f'Province: {sel_province}-> {sel_IDs[sel_id_idx]}, Stations: {sel_id_idx+1}/{len(sel_IDs)}')
            for i in range(5):
                result_t = result_f1_acc(method_to, np.array(score), df_ID[f'heavy_{i}'].to_numpy())
                result_t['province'] = sel_province
                result_t['station'] = sel_station
                result_t['stationID'] = sel_IDs[sel_id_idx]
                result_t['threshold'] = f'TH_{i}'
                results = pd.concat([results, result_t])

    results.to_csv(f'{save_to_dir}/revised_thresholds_{method}.csv', index=None)
    return results