import os,sys
import re
import nltk
import BnLemma as lm
import pandas
import string, re
import time
import torch
import random
import numpy
import pickle

from rest_framework.decorators import api_view
from rest_framework.response import Response
from sentence_transformers import SentenceTransformer
from transformers import logging, AutoTokenizer, AutoModel
from numpy import dot
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import jaccard
from sklearn.metrics.pairwise import cosine_similarity



file_path = 'stopwords.txt'

# Check if the file exists
if not os.path.isfile(file_path):

    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stop_words = stopwords.words('bengali')
    # Save stopwords to a file with UTF-8 encoding
    with open('stopwords.txt', 'w', encoding='utf-8') as file:
        for word in stop_words:
            file.write(word + '\n')


bengali_stop_words = []
with open('stopwords.txt', 'r', encoding='utf-8') as file:
    bengali_stop_words = [line.strip() for line in file]



model_name = "sagorsarker/bangla-bert-base"
tokenizer_filename = "bangla_bert_tokenizer"
model_filename = "bangla_bert_model"


if not os.path.isdir(tokenizer_filename) or not os.path.isdir(model_filename):

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # Save the tokenizer and model to files
    tokenizer.save_pretrained(tokenizer_filename)
    model.save_pretrained(model_filename)


# Load the tokenizer and model from the saved files
tokenizer = AutoTokenizer.from_pretrained(tokenizer_filename)
model = AutoModel.from_pretrained(model_filename)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


null_key = "-2398h23end7h1en198e01-sdfn9304r4t0SADSAD@DE(*@&@*@)"

# **Remove punctuation**

def f1(corpus_sentences):
    temp = []
    punctuation_pattern = r'[,:;\'"।!?.]'
    for sentence in corpus_sentences:
        clean_sentence = re.sub(punctuation_pattern, '', sentence)
        if len(clean_sentence.strip()) > 0:
            temp.append(clean_sentence)
        else:
            temp.append(null_key)
    return temp

# **Remove non Bangla word**

def f2(corpus_sentences):
    bangla_sentences = []
    for i in range(len(corpus_sentences)):
        if corpus_sentences == null_key:
            bangla_sentences.append(null_key)
            continue
        if len(corpus_sentences[i]) > 0:
            new_x = ""
            for j in range(len(corpus_sentences[i])):
                unicode_val = ord(corpus_sentences[i][j])
                if (0x0980 <= unicode_val <= 0x09FF):
                    new_x += corpus_sentences[i][j]
                elif corpus_sentences[i][j] == ' ':
                    if len(new_x) > 0 and new_x[len(new_x)-1] != ' ':
                        new_x += corpus_sentences[i][j]

            if len(new_x) > 0:
                new_x = new_x.lstrip(" ")
                new_x = new_x.rstrip(" ")
                bangla_sentences.append(new_x)
            else:
                bangla_sentences.append(null_key)
    return bangla_sentences

# **Stopword Removal**

def f3(corpus_sentences):
    temp = []
    for sentence in corpus_sentences:
        if sentence == null_key:
            temp.append(null_key)
            continue
        words = sentence.split()
        filtered_words = [word for word in words if word not in bengali_stop_words]
        filtered_sentence = ' '.join(filtered_words)
        if len(filtered_sentence.strip()) > 0:
            temp.append(filtered_sentence)
        else:
            temp.append(null_key)

    return temp

# **Root Verb**


def f4(corpus_sentences):
  bl = lm.Lemmatizer()
  temp = []
  for sentence in corpus_sentences:
      sentence = bl.lemma(sentence)
      temp.append(sentence)

  return temp

# **Acronym Expansion (UN -> United Nation)**


def expand_acronyms(sentence):
    # Define a dictionary of acronyms and their expansions
    acronym_dict = {
        "সাঃ": "সহকারী",
        "বিটিভি": "বাংলাদেশ টেলিভিশন",
        "ঢাবি": "ঢাকা বিশ্ববিদ্যালয়",
        "বেবিচক": "বেসামরিক বিমান চলাচল কর্তৃপক্ষ",
    }

    # Split the sentence into words
    words = sentence.split()

    # Iterate through the words and expand acronyms
    for i in range(len(words)):
        if words[i] in acronym_dict:
            words[i] = acronym_dict[words[i]]

    # Join the words back into a sentence
    expanded_sentence = ' '.join(words)

    return expanded_sentence

def f5(corpus_sentences):
  temp = []
  for sentence in corpus_sentences:
      expanded_sentence = expand_acronyms(sentence)
      temp.append(expanded_sentence)
  return temp


# **Multiword grouping (United Nation -> United_Nation)**


def group_multiword(sentence):
    # Define your multiword combinations and their replacements
    multiword_combinations = {
        "ঢাকা বিশ্ববিদ্যালয়": "ঢাকা_বিশ্ববিদ্যালয়",
        "বেসামরিক বিমান চলাচল কর্তৃপক্ষ": "বেসামরিক_বিমান_চলাচল_কর্তৃপক্ষ",
        # Add more multiword combinations as needed
    }

    # Iterate through multiword combinations and replace them in the sentence
    for mw, replacement in multiword_combinations.items():
        sentence = sentence.replace(mw, replacement)

    return sentence

def f6(corpus_sentences):
  temp = []
  for sentence in corpus_sentences:
      sentence = group_multiword(sentence)
      temp.append(sentence)
  return temp




def preprocessing(sentences):
    bit = 6
    i = 59
    for j in range(bit):  # Goes through each bit
        if i & (1 << j):  # Checks if the j-th bit is set in i
            if j == 0:
                sentences = f1(sentences)  # Applies function f1
                # print(sentences)
            elif j == 1:
                sentences = f2(sentences)  # Applies function f2
                # print(sentences)
            elif j == 2:
                sentences = f3(sentences)  # Applies function f3
                # print(sentences)
            elif j == 3:
                sentences = f4(sentences)  # Applies function f4
                # print(sentences)
            elif j == 4:
                sentences = f5(sentences)  # Applies function f5
                # print(sentences)
            elif j == 5:
                sentences = f6(sentences)  # Applies function f6
                # print(sentences)
    return generate_embeddings(sentences)
    

def generate_embeddings(sentences):
    
    batch_size = 1
    max_length = 512
    main_emb = []

    # Generating embeddings
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        inputs_list = [tokenizer(s, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device) for s in batch_sentences]
        input_ids = torch.cat([x['input_ids'] for x in inputs_list], dim=0)
        outputs = model(input_ids=input_ids)
        token_embeddings = outputs.last_hidden_state
        sentence_embeddings = token_embeddings.mean(dim=1)
        main_emb.extend(sentence_embeddings.cpu().detach().numpy())

        # Clear the GPU memory to prevent overflow
        del inputs_list, input_ids, outputs, token_embeddings, sentence_embeddings
        torch.cuda.empty_cache()

    return main_emb





@api_view(['POST'])
def check_plagiarism(request):
    try:
        if request.method == 'POST':
            data = request.data
            corpus_text = data.get('corpus_text', '')
            plagiarised_text = data.get('plagiarised_text', '')

            sentence_pattern = r"([।!?])"

            # corpus splitting into sentences
            corpus_text = re.split(sentence_pattern, corpus_text)
            corpus_text = [corpus_text[i] + corpus_text[i+1] for i in range(0, len(corpus_text)-1, 2)]
            temp = []
            for i in range(len(corpus_text)):
                if len(corpus_text[i]) > 1:
                    temp.append(corpus_text[i])
            corpus_text = temp

            # plagiarism splitting into sentences
            plagiarised_text = re.split(sentence_pattern, plagiarised_text)
            plagiarised_text = [plagiarised_text[i] + plagiarised_text[i+1] for i in range(0, len(plagiarised_text)-1, 2)]
            temp = []
            for i in range(len(plagiarised_text)):
                if len(plagiarised_text[i]) > 1:
                    temp.append(plagiarised_text[i])
            plagiarised_text = temp


            corpus_emb = preprocessing(corpus_text)
            plag_emb = preprocessing(plagiarised_text)


            
            max_scores = []
            max_idx = []
            tot_plag = 0
            for i_embedding in corpus_emb:
                max_similarity = -100000000
                idx = -1

                for ii, j_embedding in enumerate(plag_emb):
                    i_2d = i_embedding.reshape(1, -1)
                    j_2d = j_embedding.reshape(1, -1)
                    similarity = cosine_similarity(i_2d, j_2d)[0][0]

                    if similarity > max_similarity:
                        max_similarity = similarity
                        idx = ii

                max_scores.append(max_similarity)
                max_idx.append(idx)

                if max_similarity >= 0.80:
                    tot_plag += 1

            score = (tot_plag/len(corpus_text))*100.0
            print(score)

            temp = []
            for idx, sentence in enumerate(corpus_text):
                if max_scores[idx] < 0.8:
                    continue
                temp.append({
                    "score": score,
                    "corpus_sentence": corpus_text[idx],
                    "plagiarism_sentence": plagiarised_text[max_idx[idx]],
                })

            return Response(temp)
    except Exception as e:
        return Response({"error": str(e)}, status=500)
