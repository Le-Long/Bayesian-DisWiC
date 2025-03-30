import cloudpickle
import itertools
import argparse
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import scipy.stats as stats
from tqdm import tqdm
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Case 1: Use concantenated embeddings as features  ##
def concat(dataframes, file_names)
    embeddings_lists = [[], []]

    # retrieve the context embeddings using the identifiers from the dataframe
    for df, file_name, embeddings in zip(dataframes, file_names, embeddings_lists ):
        loaded_embeddings = np.load(file_name)
        for _, row in tqdm(df.iterrows()):
            try:
                context_embedding1 = loaded_embeddings[row['identifier1']]
                context_embedding2 = loaded_embeddings[row['identifier2']]
                # concatenate the embeddings to form a single feature vector
                concatenated_emb = np.concatenate((context_embedding1, context_embedding2))
                embeddings.append(concatenated_emb)
            except KeyError as e:
                print(f"KeyError: {e}. Identifier not found in embeddings file.")
                embeddings.append(np.nan)
                continue

    # convert the lists of feature vectors to numpy arrays (feature matrices)
    train_embeddings = np.array(embeddings_lists[0])
    dev_embeddings = np.array(embeddings_lists[1])
    # Downrank the embeddings with PCA
    pca = PCA(n_components=n_pca)

    train_embeddings = StandardScaler().fit_transform(train_embeddings)
    df_train_set['concate_pca'] = pca.fit_transform(train_embeddings).tolist() 
    dev_embeddings = StandardScaler().fit_transform(dev_embeddings)
    df_dev_set['concate_pca'] = pca.fit_transform(dev_embeddings).tolist()

    df_train_set = df_train_set[~df_train_set['concate_pca'].isnull()]
    df_dev_set = df_dev_set[~df_dev_set['concate_pca'].isnull()]

    return df_train_set, df_dev_set


# Case 2: Use cosine similarity as features ##
def cosine(dataframes, file_names)
    from sklearn.metrics.pairwise import cosine_similarity

    dataframes = [df_dev_set, df_train_set]
    file_names = ['data/dev_embeddings.npz', 'data/train_embeddings.npz']

    print(len(df_train_set), len(df_dev_set))
    cosine_similarities_lists = [[], []]

    # iterate over the lists to compute and store cosine similarities
    for df, file_name, cosine_similarities in zip(dataframes, file_names, cosine_similarities_lists):
        loaded_embeddings = np.load(file_name)
        for _, row in tqdm(df.iterrows()):
            try:
                context_embedding1 = loaded_embeddings[row['identifier1']] 
                context_embedding2 = loaded_embeddings[row['identifier2']]
                cosine_sim = cosine_similarity([context_embedding1], [context_embedding2])[0][0]
                cosine_similarities.append(cosine_sim)
            except KeyError as e:
                # print(f"KeyError: {e}. row not there")
                cosine_similarities.append(np.nan)
                continue
        # add the cosine similarities to the dataFrame
        df['cosine_similarity'] = cosine_similarities

    df_train_set = df_train_set[~df_train_set['cosine_similarity'].isnull()]
    df_dev_set = df_dev_set[~df_dev_set['cosine_similarity'].isnull()]
    return df_train_set, df_dev_set


def mean_abs_disagreement_func(x):
    return np.nan_to_num(np.nanmean([abs(pair[1] - pair[0]) for pair in itertools.combinations(list(x),2)]), nan=0.0)


def main(features, val=True):
    # Loading pairwise datasets
    df_train_set = pd.read_csv('data/train_judgments.csv')
    df_dev_set = pd.read_csv('data/dev_judgments.csv')
    # df_train_set['concate_pca'] = df_train_set['concate_pca'].apply(literal_eval)
    # df_dev_set['concate_pca'] = df_dev_set['concate_pca'].apply(literal_eval)

    dataframes = [df_train_set, df_dev_set]
    file_names = ['data/train_embeddings.npz', 'data/dev_embeddings.npz']
    if features: 
        n_pca = 8
        df_train_set, df_dev_set = concat(dataframes, file_names)
    else:
        n_pca = 1
        df_train_set, df_dev_set = cosine(dataframes, file_names)

    # save new features to files
    df_train_set['median_judgment'] = df_train_set['median_judgment'].astype(float)
    df_train_set['mean_disagreement'] = df_train_set['mean_disagreement'].astype(float)
    print(len(df_train_set), len(df_dev_set))
    df_train_set.to_csv('data/train_judgments_clean.csv', index=False)
    df_dev_set.to_csv('data/dev_judgments_clean.csv', index=False)

    # define linear model and fit the posterior
    array = np.array(df_train_set['concate_pca'].tolist()) if features else df_train_set['cosine_similarity']
    xbar = np.mean(array, axis=0)
    with pm.Model(coords={'features':[i for i in range(n_pca)]}) as WiC:
        # priors
        a = pm.Normal("a", mu=1, sigma=3)
        b = pm.Normal("b", mu=0, sigma=1, dims="features")
        sigma = pm.Uniform("sigma", 0, 1)
        
        # observed data
        xdata = pm.Data('xdata', array - xbar, mutable=True)
        mu = a + xdata.dot(b) if features else a + b * xdata
        proximity = pm.Normal("proximity", mu=mu, sigma=sigma, observed=df_train_set['median_judgment'])

        # fitting
        trace_wic = pm.sample(200, tune=10, cores=1, chains=4)

    # # # save the model
    pickle_filepath = f'pickle_concate.pkl'
    dict_to_save = {'model': WiC,
                    'trace': trace_wic,
                    }

    with open(pickle_filepath , 'wb') as buff:
        cloudpickle.dump(dict_to_save, buff)

    # with open(pickle_filepath , 'rb') as buff:
    #     model_dict = cloudpickle.load(buff)

    # trace_wic = model_dict['trace']
    # WiC = model_dict['model']

    # # in-sample predictions to test on training data
    with WiC:
        judgments_pred = pm.sample_posterior_predictive(trace_wic)
        judgments_pred = az.extract(judgments_pred, num_samples=10, group='posterior_predictive')
        mean_abs_disagreement = judgments_pred.reduce(func=(lambda data, axis: np.apply_along_axis(mean_abs_disagreement_func, axis, data)), dim='sample')
        res = stats.spearmanr(mean_abs_disagreement.to_pandas()['proximity'], df_train_set['mean_disagreement'])
        print('train set rho score: ', res.statistic)

    if val:
        # sample and evaluate on the dev set
        array = np.array(df_dev_set['concate_pca'].tolist())
        xbar = np.mean(array, axis=0)
        with pm.Model(coords={'features':[i for i in range(n_pca)]}) as WiC_predict:
            # priors
            a = pm.Normal("a", mu=1, sigma=3)
            b = pm.Normal("b", mu=0, sigma=1, dims="features")
            sigma = pm.Uniform("sigma", 0, 1)
            
            # observed data
            xdata = pm.Data('xdata', array - xbar, mutable=True)
            mu = a + xdata.dot(b)
            print(mu.shape.eval())
            proximity = pm.Normal("proximity", mu=mu, sigma=sigma)

            # out-of-sample predictions
            predictions = pm.sample_posterior_predictive(trace_wic, predictions=True, var_names=['proximity'])
            predictions = az.extract(predictions, num_samples=10, group='predictions')
            mean_abs_disagreement = predictions.reduce(func=(lambda data, axis: np.apply_along_axis(mean_abs_disagreement_func, axis, data)), dim='sample')
            res = stats.spearmanr(mean_abs_disagreement.to_pandas()['proximity'], df_dev_set['mean_disagreement'])
            print('dev set rho score: ', res.statistic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, help="choose cosine or concat")
    args = parser.parse_args()

    if args.features == "concat":
        main(features=True)
    elif args.features == "cosine":
        main(features=False)
    else:
        print("Wrong type of features. You must choose either cosine or concat")
