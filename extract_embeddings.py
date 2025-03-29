import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, XLMRobertaModel

##### Approriate from baselines.ipynb from coda #####

label_file_paths_train = 'data/train_judgments.csv'
uses_file_paths_train = 'data/all_uses.csv'
label_file_paths_dev = 'data/dev_judgments.csv'
uses_file_paths_dev = 'data/all_uses.csv'

# loading train labels and uses and dev instances and uses

# Again, uses_list should be split between train and dev
train_set = pd.read_csv(label_file_paths_train).to_dict('records')
train_uses_list = pd.read_csv(uses_file_paths_train).to_dict('records')
dev_uses_list = pd.read_csv(uses_file_paths_dev).to_dict('records')
dev_set = pd.read_csv(label_file_paths_dev).to_dict('records')


# make dictionaries to map identifiers to their contexts and target token indices from the train and dev uses data


def create_mappings(uses_list):
    id2context = {}
    id2idx = {}
    for row in uses_list:
        identifier = row['identifier']
        context = row['context']
        idx = row['indexes_target_token']
        id2context[identifier] = context
        id2idx[identifier] = idx
    return id2context, id2idx


train_id2context, train_id2idx = create_mappings(train_uses_list)
dev_id2context, dev_id2idx = create_mappings(dev_uses_list)


# merging train labels and uses into a single dataframe
def merge_dataset(data_set, id2context, id2idx):
    data_uses_merged = []
    for row in data_set:
        identifier1 = row['identifier1']
        identifier2 = row['identifier2']

        # use id2context dictionary to get the corresponding context for each identifier
        context1 = id2context.get(identifier1)
        context2 = id2context.get(identifier2)

        if context1 is None and context2 is None:
            continue

        # use id2idx dictionary to get the corresponding target token index for each identifier
        index_target_token1 = id2idx.get(identifier1)
        index_target_token2 = id2idx.get(identifier2)

        lemma = row['lemma']
        mean_disagreement = row['mean_disagreement']
        median_judgment = row['median_judgment']
        # judgments = row['judgments']
        language = row['language']
        data_row = {'context1': context1, 'context2': context2, 'index_target_token1': index_target_token1,
                    'index_target_token2': index_target_token2,
                    'identifier1': identifier1, 'identifier2': identifier2, 'lemma': lemma,
                    'mean_disagreement': mean_disagreement, 'median_judgment': median_judgment, 'language': language}

        data_uses_merged.append(data_row)
    return data_uses_merged


df_train_uses_merged = pd.DataFrame(merge_dataset(train_set, train_id2context, train_id2idx))
df_dev_uses_merged = pd.DataFrame(merge_dataset(dev_set, dev_id2context, dev_id2idx))
print(df_train_uses_merged.shape, df_dev_uses_merged.shape)

# define and load the tokenizer and model for XL-lexeme and XLM-RoBERTa
# uncomment when use XL-lexeme
# from WordTransformer import WordTransformer, InputExample
# model = WordTransformer('pierluigic/xl-lexeme')
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
model = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# truncate the input to max 512
def truncation_indices(target_subword_indices: list[bool], truncation_tokens_before_target=0.5) -> tuple[int, int]:
    max_tokens = 512
    n_target_subtokens = target_subword_indices.count(True)
    tokens_before = int((max_tokens - n_target_subtokens) * truncation_tokens_before_target)
    tokens_after = max_tokens - tokens_before - n_target_subtokens

    # get index of the first target subword
    lindex_target = target_subword_indices.index(True)
    # get index of the last target subword
    rindex_target = lindex_target + n_target_subtokens
    # starting index for truncation
    lindex = max(lindex_target - tokens_before, 0)
    # ending index for truncation
    rindex = rindex_target + tokens_after
    return lindex, rindex


# get matrices of features
def get_target_token_embedding(context, index, truncation_tokens_before_target=0.5, lexeme=False):
    max_tokens = 512
    start_idx = int(str(index).strip().split(':')[0])
    end_idx = int(str(index).strip().split(':')[1])

    if lexeme:
        if len(context) > max_tokens * 5:
            # simple truncate without tokenization
            inputs = InputExample(texts=row['context1'][:2500], positions=[start_idx, end_idx])
        else:
            inputs = InputExample(texts=row['context1'], positions=[start_idx, end_idx])
        return model.encode(inputs)

    # tokenize the context with offset mapping
    inputs = tokenizer(context, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)

    # offset mapping to provide the start and end positions of each token in the original context
    offset_mapping = inputs['offset_mapping'][0].tolist()

    # convert input ids to tokens
    input_ids = inputs['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # create a boolean mask for subwords within the target words span
    subwords_bool_mask = [
        (start <= start_idx < end) or (start < end_idx <= end) or (start_idx <= start and end <= end_idx)
        for start, end in offset_mapping
    ]

    target_token_indices = [i for i, value in enumerate(subwords_bool_mask) if value]

    if not target_token_indices:
        print(
            f"Error: Target token indices not found within the specified range for context: '{context}' and index: '{index}'")
        return None

    # truncate input if it exceeds 512 tokens
    if len(input_ids[0]) > max_tokens:
        # truncation indices based on the subwords boolean mask
        lindex, rindex = truncation_indices(subwords_bool_mask, truncation_tokens_before_target)

        # truncate the tokens, input_ids and subwords_bool_mask within the range of truncation indices
        tokens = tokens[lindex:rindex]
        context = ' '.join(tokens)
        input_ids = input_ids[:, lindex:rindex]
        subwords_bool_mask = subwords_bool_mask[lindex:rindex]
        offset_mapping = offset_mapping[lindex:rindex]
        inputs['input_ids'] = input_ids  # update the input_ids in the inputs dictionary

        # check if truncation was successful
        if len(input_ids[0]) > max_tokens:
            print(
                f"Truncation failed: input sequence length ({len(input_ids[0])}) exceeds the maximum token limit for context: '{context}' and index: '{index}'")
            return None

    # extract the subwords for the target word
    extracted_subwords = [tokens[i] for i, value in enumerate(subwords_bool_mask) if value]

    if not extracted_subwords:
        print(f"Error: no subwords extracted for the target word in context: '{context}' and index: '{index}'")
        return None

    with torch.no_grad():
        outputs = model(inputs['input_ids'].to(device))  # get embeddings for the truncated input

    # embeddings for all tokens in the truncated input
    embeddings = outputs.last_hidden_state[0].to('cpu')

    # embeddings for target token
    target_embeddings = embeddings[subwords_bool_mask]

    if target_embeddings.size(0) == 0:
        print(f"error: no embeddings found for the target token in context: '{context}' and index: '{index}'")
        return None

    # aggregated target token embedding
    target_embeddings_nump = target_embeddings.mean(dim=0).numpy()

    return target_embeddings_nump


# getting target token embeddings for contexts in train and dev 
dataframes = [df_train_uses_merged, df_dev_uses_merged]
file_names = ['train_embeddings.npz', 'dev_embeddings.npz']

for df, file_name in zip(dataframes, file_names):
    id2embedding = {}
    print(df.shape)

    for i, row in tqdm(df.iterrows(), position=0, leave=True):
        identifier1 = row['identifier1']
        identifier2 = row['identifier2']

        if identifier1 not in id2embedding:
            embedding1 = get_target_token_embedding(row['context1'], row['index_target_token1'])
            id2embedding[identifier1] = embedding1

        if identifier2 not in id2embedding:
            embedding2 = get_target_token_embedding(row['context2'], row['index_target_token2'])
            id2embedding[identifier2] = embedding2

        if i % 2000 == 0:
            np.savez(file_name, **id2embedding)

    # store embeddings in a .npz file using identifiers as keys
    np.savez(file_name, **id2embedding)
