# Bayesian model of Disagreement in Word-in-Context Ranking

## Requirements

After clone this directory to your machine, you can install pymc and other libraries:

```bash
cd Bayesian-DisWiC/
python -m pip install requirements.txt
```

### Data



#### Data Preparation

To create these files:
1. 
2. 



### Fitting



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


## Citation

A large part of the python files are based on the code provided by the organizer from [the shared task at the CoMeDi workshop](https://comedinlp.github.io/#task). The analyze jupyter notebook is based on the notebook of Chapter 4 for the course [Statistical rethinking (second edition) with python and pymc](https://github.com/pymc-devs/pymc-resources/tree/main/Rethinking_2). We thank the authors for releasing these codes

