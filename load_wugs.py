import requests
import os
import zipfile
import shutil
import pandas as pd
from pathlib import Path
import unicodedata
import itertools
import numpy as np
from sklearn.model_selection import train_test_split

##### Approriate from load_wugs.ipynb #####

## Download data ##
datasets_to_download_zenodo = [('refwug', 'German', '1.1.0', 'https://zenodo.org/records/5791269/files/refwug.zip?download=1', ''), 
                            ('durel', 'German', '3.0.0', 'https://zenodo.org/records/5784453/files/durel.zip?download=1', ''),
                            ('surel', 'German', '3.0.0', 'https://zenodo.org/records/5784569/files/surel.zip?download=1', ''), 
                            ('chiwug', 'Chinese', '1.0.0', 'https://zenodo.org/records/10023263/files/chiwug.zip?download=1', ''),
                            ('dwug_de', 'German', '3.0.0', 'https://zenodo.org/records/14028509/files/dwug_de.zip?download=1', ''),
                            ('dwug_en', 'English', '3.0.0', 'https://zenodo.org/records/14028531/files/dwug_en.zip?download=1', ''),
                            ('dwug_sv', 'Swedish', '3.0.0', 'https://zenodo.org/records/14028906/files/dwug_sv.zip?download=1', ''),
                            ('dwug_de_resampled', 'German', '1.0.0', 'https://zenodo.org/records/12670698/files/dwug_de_resampled.zip?download=1', ''),
                            ('dwug_en_resampled', 'English', '1.0.0', 'https://zenodo.org/records/14025941/files/dwug_en_resampled.zip?download=1', ''),
                            ('dwug_sv_resampled', 'Swedish', '1.0.0', 'https://zenodo.org/records/14026615/files/dwug_sv_resampled.zip?download=1', ''),
                            ('discowug', 'German', '2.0.0', 'https://zenodo.org/records/14028592/files/discowug.zip?download=1', ''),
                            ('dwug_es', 'Spanish', '4.0.0', 'https://zenodo.org/records/6433667/files/dwug_es.zip?download=1', ''),
                                ]

datasets_to_download_github = [('rusemshift_1', 'Russian', '', 'https://github.com/juliarodina/RuSemShift/archive/refs/heads/master.zip', 'RuSemShift-master/rusemshift_1/DWUG/data/'), 
                                ('rusemshift_2', 'Russian', '', 'https://github.com/juliarodina/RuSemShift/archive/refs/heads/master.zip', 'RuSemShift-master/rusemshift_2/DWUG/data/'), 
                                ('rushifteval1', 'Russian', '', 'https://github.com/akutuzov/rushifteval_public/archive/refs/heads/main.zip', 'rushifteval_public-main/durel/rushifteval1/data/'), 
                                ('rushifteval2', 'Russian', '', 'https://github.com/akutuzov/rushifteval_public/archive/refs/heads/main.zip', 'rushifteval_public-main/durel/rushifteval2/data/'), 
                                ('rushifteval3', 'Russian', '', 'https://github.com/akutuzov/rushifteval_public/archive/refs/heads/main.zip', 'rushifteval_public-main/durel/rushifteval3/data/'), 
                                ('rudsi', 'Russian', '', 'https://github.com/kategavrishina/RuDSI/archive/refs/heads/main.zip', 'RuDSI-main/data/'), 
                                ('nordiachange1', 'Norwegian', '', 'https://github.com/ltgoslo/nor_dia_change/archive/refs/heads/main.zip', 'nor_dia_change-main/subset1/data/'), 
                                ('nordiachange2', 'Norwegian', '', 'https://github.com/ltgoslo/nor_dia_change/archive/refs/heads/main.zip', 'nor_dia_change-main/subset2/data/')
                              ]

datasets_all = datasets_to_download_zenodo + datasets_to_download_github

if not os.path.exists('data/'):
    os.makedirs('data/')      

    for name, language, version, link, path_to_data in datasets_to_download_zenodo:
        r = requests.get(link, allow_redirects=True)
        f = 'data/' + name + '.zip'
        open(f, 'wb').write(r.content)

    for name, language, version, link, path_to_data in datasets_to_download_github:
        r = requests.get(link, allow_redirects=True)
        f = 'data/' + name + '.zip'
        open(f, 'wb').write(r.content)
    
    # Unzip data and remove superfluous files ##
    for name, language, version, link, path_to_data in datasets_all:
        if not os.path.exists('data/' + name):
            os.makedirs('data/' + name)
        else:
            shutil.rmtree('data/' + name)        
            os.makedirs('data/' + name)
        if path_to_data == '':
            with zipfile.ZipFile('data/' + name + '.zip') as z:
                z.extractall('data/temp')
            dest = shutil.move('data/temp/' + name + '/data', 'data/' + name)  
        else:
            with zipfile.ZipFile('data/' + name + '.zip') as z:
                z.extractall('data/temp/' + name)
            dest = shutil.move('data/temp/' + name + '/' + path_to_data, 'data/' + name + '/data')  
        shutil.rmtree('data/temp/' + name)  

languages_global = ['German', 'English', 'Swedish', 'Spanish', 'Chinese', 'Russian', 'Norwegian']
 
# Load datasets into data frame
df_judgments = pd.DataFrame()
j = 0
i2lemma2name_judgments = []
for name, language, version, link, path_to_data in datasets_all:
    i = 0
    for p in Path('data/'+name+'/data').glob('*/judgments.csv'):
        lemma = str(p).split('/')[-2]        
        lemma = unicodedata.normalize('NFC', lemma)
        df = pd.read_csv(p, delimiter='\t', quoting=3, na_filter=False)
        df['dataset'] = name
        df['language'] = language
        df['annotator'] = df['annotator'].astype(str) + '-' + name # make sure annotators are unique across datasets
        if name in ['chiwug']:            
            df['identifier1'] = df['identifier1'].astype(str) + '-' + str(i) # make sure identifiers are unique across words
            df['identifier2'] = df['identifier2'].astype(str) + '-' + str(i) # make sure identifiers are unique across words
        if name in ['rusemshift_1', 'rusemshift_2']: # only done four judgments for those datasets which will not be mapped later
            # don't do this for the German data where same identifiers mean same use
            df['identifier1'] = df['identifier1'].astype(str) + '-' + str(j) # make sure identifiers are unique across datasets
            df['identifier2'] = df['identifier2'].astype(str) + '-' + str(j) # make sure identifiers are unique across datasets
        df['judgment'] = df['judgment'].astype(float)
        df_judgments = pd.concat([df_judgments, df])
        i2lemma2name_judgments.append((i,lemma,name))
        i+=1
    j+=1
    
df_uses = pd.DataFrame()
j = 0
i2lemma2name_uses = []
for name, language, version, link, path_to_data in datasets_all:
    i = 0
    for p in Path('data/'+name+'/data').glob('*/uses.csv'):
        lemma = str(p).split('/')[-2]        
        lemma = unicodedata.normalize('NFC', lemma)
        if name in ['rushifteval1', 'rushifteval2', 'rushifteval3', 'nordiachange1', 'nordiachange2']:
            df = pd.read_csv(p, delimiter='\t', quoting=0, quotechar='"', engine="python", on_bad_lines='skip' , na_filter=False)
        else:        
            df = pd.read_csv(p, delimiter='\t', quoting=3, na_filter=False)
        df['dataset'] = name
        df['language'] = language
        if name in ['chiwug']:
            df['identifier'] = df['identifier'].astype(str) + '-' + str(i) # make sure identifiers are unique across words
            df['lemma'] = df['lemma'].apply(lambda x: unicodedata.normalize('NFC', x))
        if name in ['rushifteval1', 'rushifteval2', 'rushifteval3', 'rusemshift_1', 'rusemshift_2']:
            df['identifier'] = df['identifier'].astype(str) + '-' + str(j) # make sure identifiers are unique across datasets
        df_uses = pd.concat([df_uses, df])        
        i2lemma2name_uses.append((i,lemma,name))
        i+=1
    j+=1

assert i2lemma2name_judgments == i2lemma2name_uses
assert not 'nan' in df_judgments['identifier1'].astype(str).unique()
assert not 'nan' in df_judgments['identifier2'].astype(str).unique()
df_judgments_length_before_sorting = len(df_judgments)
df_judgments[['identifier1','identifier2']] = np.sort(df_judgments[['identifier1','identifier2']], axis=1) # sort within pairs to be able to aggregate
assert df_judgments_length_before_sorting == len(df_judgments)

## Clean and aggregate data ##
# Replace 0.0 judgments with nan
df = df_judgments.copy()
df['judgment'] = df['judgment'].replace(0.0, np.NaN)

# Aggregate use pairs and extract median column
df = df.groupby(['identifier1', 'identifier2', 'lemma', 'dataset', 'language'])['judgment'].apply(list).reset_index(name='judgments')
df['median_judgment'] = df['judgments'].apply(lambda x: np.nanmedian(list(x)))
df['mean_disagreement'] = df['judgments'].apply(lambda x: np.nan_to_num(np.nanmean([abs(pair[1] - pair[0]) for pair in itertools.combinations(list(x),2)]), nan=0.0))

# Remove pairs with nan median
df = df[~df['median_judgment'].isnull()]
df_judgments_aggregated = df.copy()

# Display a sample to validate
print(df_judgments_aggregated.head(5))
df_uses = df_uses[['identifier', 'lemma', 'dataset', 'context', 'indexes_target_token', 'language']]
print(df_uses.head(5))

# split train dev set and save them
train_judgments, dev_judgments = train_test_split(df_judgments_aggregated, test_size=0.2) 
# NOTE: should also split the uses file
train_judgments.to_csv('data/train_judgments.csv', index=False)  
dev_judgments.to_csv('data/dev_judgments.csv', index=False)  
df_uses.to_csv('data/all_uses.csv', index=False)