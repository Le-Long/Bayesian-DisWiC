# Bayesian model of Disagreement in Word-in-Context Ranking

This repo contains the code to download the data from Word Usage Graphs (WUGs), fit a probabilistic linear model to the median annotated semantic proximity, and then predict the level of annotator disagreement with posterior sampling. After fitting the model, you can also play with the data using the `analyze.ipynb` notebook.

## Requirements

After clone this directory to your machine, you can install pymc and other libraries:

```bash
cd Bayesian-DisWiC/
python -m pip install requirements.txt
```

### Data

The annotators judgments and word-in-context uses data has already been splited into train and dev in the `data` folder, you only need to follow the second step. If you still want to create your own dataset, just remove the `data` folder or rename it and then follow these instructions   

#### Data Preparation

1. Downloading all data to a new `data` folder, calculating the median labels and the mean disagreement:
```bash
python load_wugs.py
```
2. From the dataset, calculating the contextual embedding matrices (currently using XLM-RoBERTa-base, you can edit the code to use other model like XL-lexeme)
```bash
python extract_embeddings.py
```


### Fitting

With the data prepared, we want to represent each pair of word usages with a variable. You can choose to concatenate the two embeddings or use the cosine similarity, then take them as the input for a probabilistic linear model. The Pearson rho will be measured between the true annotators disagreement and the one computed from posterior predictive samples of the fitted model.

#### Running

Use the following command to start fitting the model and evaluate it on train set and dev set:

```bash
python model.py \
        --features concat
```

**Command Options:**
- `--features`: Choosing how to represent a pair of word usages from contextual embeddings. 
Option: `concat` - concatenating the two vectors and use PCA to shorten the new vector
        `cosine` - compute cosine similarity between two vectors


## Acknowledgement

A large part of the python files are based on the code provided by the organizer from [the shared task at the CoMeDi workshop](https://comedinlp.github.io/#task). The analyze jupyter notebook is based on the notebook of Chapter 4 for the course [Statistical rethinking (second edition) with python and pymc](https://github.com/pymc-devs/pymc-resources/tree/main/Rethinking_2). We thank the authors for releasing these codes

