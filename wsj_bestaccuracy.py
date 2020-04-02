# -*- coding: utf-8 -*-
"""
Authors: Pranav Krishna, Tejasvi Navnage, Ganesh Nalluru, Alagappan

Functions:
    load_files- Loads the training and test files and coverts into feature 
    and target dataframe.
    
    Bidirectional_LSTM-The function returns a bidirectional LSTM model with a Bi-LSTM layer 
    for each of the four inputs and one extra Bi-LSTM layer for sentences.
    
    Embed- Converts words into embeding matix
    
    vectorize_line-Converts words to index

Algorithm:
    The feature dataframe is converted into index. 
    Glove embeddings are used for word to vector conversions.
    The target y variable is converted into one hot encoded form.
    All the 4 features-Parts of speech, Named entities, sentences and verbs
    are vectorized and sent to the stacked Bi-LSTM model.
    Output vectors are converted back to tags and the output is evaluvated
    using coneval-2003 script

Model Selection:
    The Bi-LSTM model architecture is finalized after various other models like encoder-decoder,
    XGBoost.
    Since Keras requires all the layers to be concatenated to have same dimension, the verb is 
    repeated max-senetence-length times and fed as input.
    The current architecture gave better accuracy than using muliple layers after
    concatenating.
"""
import codecs
import os
import sys
import pandas as pd
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.optimizers import Adam

from gensim import models
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM,TimeDistributed,Bidirectional, InputLayer, RepeatVector
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate,Dot
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np
import re
from keras import backend as K

#from attention_decoder import AttentionDecoder

input_data_path=r'train-set.txt'
input_test_path=r'test-set.txt'


#input_data_path = r"train-set.txt"
#input_test_path = r"test-set.txt"
l_idx = 4
p_idx = 5

pos_set = set()
verb_set = set()
vocab = set()
tag_set = set()
ne_tags_set=set()

def load_files(input_data_path):
    '''
    The following function takes input file path. It converts them into
    feature and target data frame. If there are more than one verb in a sentence
    same sentence is repeated "number of verbs in the sentence" times. Target and 
    the named entity tags are converted into BIO format.
    '''
    global verb_set
    global vocab
    global pos_set
    total_props = 0
    total_sents = 0
    all_props = []
    max_sent_len = 0
    vs = []
    prev_words = ''
    words = []
    props = []
    tags = []
    spans = []
    all_props = []
    named_entities = []
    pos = []
    partial_synt = []
    full_synt = []
    vs = []
    targets = []
    ne_tags=[]

    label_dict = {}

    sents = []
    named_entities_list = []
    pos_list = []
    targets_list = []
    full_synt_list = []
    tags_list = []
    ne_spans=""

    fin = open(input_data_path, 'r')
    for line in fin:
        line = line.strip()
        if line == '':
            joined_words = " ".join(words)
            prev_words = joined_words
            total_props += len(props)
            total_sents += 1
            propid_labels = ['O' for _ in words]
            for t in range(len(props)):
                propid_labels[props[t]] = 'V'
                sents.append(words)
                named_entities_list.append(ne_tags)
                pos_list.append(pos)
                targets_list.append(targets[t])
                full_synt_list.append(full_synt)
                tags_list.append(tags[t])
                max_sent_len = max(len(words), max_sent_len)
            words = []
            props = []
            tags = []
            spans = []
            all_props = []
            named_entities = []
            pos = []
            partial_synt = []
            full_synt = []
            vs = []
            targets = []
            ne_tags=[]
            ne_spans=""
            continue

        info = line.split()
        word = info[0]
        named_entities.append(info[3])
        pos.append(info[1])
        pos_set.add(info[1])
        full_synt.append(info[2])
        words.append(word.lower())
        vocab.add(word.lower())
        idx = len(words) - 1
        if idx == 0:
            tags = [[] for _ in info[p_idx:]]
            spans = ["" for _ in info[p_idx:]]
        is_predicate = (info[l_idx] != '-')
        ne=info[3]
        ne_label=ne.strip("()*")
        
        if "(" in ne:
                ne_tags.append("B-" + ne_label)
                ne_spans = ne_label
                ne_tags_set.add("B-" + ne_label)
        elif ne_spans != "":
                ne_tags.append("I-" + ne_spans)
                ne_tags_set.add("I-" + ne_spans)
        else:
                ne_tags.append("O")
                ne_tags_set.add("O")
        if ")" in ne:
                ne_spans = ""
        for t in range(len(tags)):
            arg = info[p_idx + t]
            label = arg.strip("()*")
            label_dict[arg] = 1
            
            if "(" in arg:
                tags[t].append("B-" + label)
                spans[t] = label
                tag_set.add("B-" + label)
            elif spans[t] != "":
                tags[t].append("I-" + spans[t])
                tag_set.add("I-" + spans[t])
            else:
                tags[t].append("O")
                tag_set.add("O")
            if ")" in arg:
                spans[t] = ""

        if is_predicate:
            props.append(idx)
            targets.append(info[4])
            verb_set.add(info[4])
            vocab.add(info[4])

    fin.close()
    # fout_propid.close()
    df = pd.DataFrame(
        {'sent': sents,
         'pos': pos_list,
         'synt': full_synt_list,
         'ne': named_entities_list,
         'verbs': targets_list,
         'target': tags_list
         })
    return df


def ignore_class_accuracy(to_ignore=0):
    #Ignore class accuracy for calculating non-majority class accuracy
    #https://nlpforhackers.io/lstm-pos-tagger-keras/
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)

        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy

    return ignore_accuracy

def Bidirectional_LSTM(max_len,embedding_matrix):
    '''
    The function returns a bidirectional LSTM model with a Bi-LSTM layer 
    for each of the four inputs and one extra Bi-LSTM layer for sentences.
    '''
    input_1 = Input(shape=(max_len,))
    embedding_layer = Embedding(len(word2index), 300,weights=[embedding_matrix])(input_1)
    LSTM_Layer_1 = Bidirectional(LSTM(512, return_sequences=True))(embedding_layer)
    dropout_1 = Dropout(0.3)(LSTM_Layer_1)
    LSTM_Layer_11 = Bidirectional(LSTM(512, return_sequences=True))(dropout_1)

    input_2 = Input(shape=(max_len,))
    embedding_layer_2 = Embedding(len(word2index), 300,weights=[embedding_matrix])(input_2)
    LSTM_Layer_2 = Bidirectional(LSTM(512, return_sequences=True))(embedding_layer_2)
    
    input_3 = Input(shape=(max_len,))
    embedding_layer_3 = Embedding(len(word2index), 300)(input_3)
    LSTM_Layer_3 = Bidirectional(LSTM(512, return_sequences=True))(embedding_layer_3)

    input_4 = Input(shape=(max_len,))
    embedding_layer_4 = Embedding(len(word2index), 300)(input_4)
    LSTM_Layer_4 = Bidirectional(LSTM(512, return_sequences=True))(embedding_layer_4)

    concat_layer = Concatenate()([LSTM_Layer_11, LSTM_Layer_2,LSTM_Layer_3,LSTM_Layer_4])
    dense_layer_3 = TimeDistributed(Dense(300, activation='relu'))(concat_layer)
    output = Dense(68, activation='softmax')(dense_layer_3)
    model = Model(inputs=[input_1,input_2,input_3,input_4], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',ignore_class_accuracy(0)])
    print(model.summary())
    return model


def embed(word2index_dict):
    # Create embedding matrix using word2vec
    embeddings_index = models.KeyedVectors.load_word2vec_format(
        'GoogleNews-vectors-negative300.bin', binary=True)
    embedding_matrix = np.zeros((len(word2index_dict), 300))
    embeddings_index['-DOCSTART-'] = np.zeros(300)
    for word, i in word2index_dict.items():
        
        if (word in embeddings_index):
            embedding_vector = embeddings_index[word]
        else:
            embedding_vector = np.zeros(300)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def vectorize_line(line_list, words):
    # Convert word to index
    word2index_vec = {w: i for i, w in enumerate(list(words))}
    word2index_vec['-PAD-'] = 0  # The special value used for padding
    word2index_vec['-OOV-'] = 1  # The special value used for OOVs
    train_list = []
    for s in line_list:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index_vec[w])
            except KeyError:
                # print(w)
                s_int.append(word2index_vec['-OOV-'])
        train_list.append(s_int)
    
    return train_list, word2index_vec


def pad_seq(line_list, max_len):
    # Pad each line
    sent_list = pad_sequences(line_list, maxlen=max_len, padding='post')
    return sent_list


def pad_tags(tags_list, max_len):
    for i in range(0, len(tags_list)):
        tags_list[i] += ['<pad>'] * (max_len - len(tags_list[i]))

    return tags_list


def vectorize_tag(tags, tag_list):
    # Convert tags to index
    tag2index = {t: i for i, t in enumerate(list(tags))}
    test_tags_y = []
    for s in tag_list:
        test_tags_y.append([tag2index[t] for t in s])
    return test_tags_y,tag2index


def to_categorical(sequences, categories):
    # Create onehot encoding of y variable(tags)
    cat_sequences = []

    for s in sequences:
        cats = []
        for item in s:
            x = np.ndarray(categories)
            x[:] = 0
            x[item] = 1.0
            cats.append(x)
        cat_sequences.append(cats)
    return np.array(cat_sequences)


def get_one_hot(sent_lists, dictvec):
    train_sentences_X = []
    for s in sent_lists:
        s_int = []
        for w in s:
            try:
                s_int.append(dictvec[w.lower()])
            except KeyError:
                s_int.append(dictvec['-OOV-'])
        train_sentences_X.append(s_int)
    return train_sentences_X




#def encoder_decoder(N_BLOCKS, N_OUTPUTS):
#    Encoder decoder model which did not workout well
#    model = Sequential()
#    model.add(InputLayer(input_shape=(MAX_LENGTH,)))
#    model.add(Embedding(len(word2index), 128,trainable=True))
#    model.add(Bidirectional(LSTM(100, return_sequences=True)))
#    model.add(Bidirectional(LSTM(100, return_sequences=False)))
#    model.add(RepeatVector(N_OUTPUTS))
#    model.add(Bidirectional(LSTM(100, return_sequences=True)))
#    #model.add(Bidirectional(LSTM(512, return_sequences=False)))
#    model.add(TimeDistributed(Dense(68)))
#    model.add(Activation('softmax'))
#    model.compile(loss='categorical_crossentropy',
#                  optimizer=Adam(0.0001),
#                  metrics=['accuracy'])
#    model.summary()
#    return model


percentile_list_train = load_files(input_data_path)  # Loading train files
percentile_list_test = load_files(input_test_path)  # Loading test files

vocab=list(vocab)
train_text_vec, word2index = vectorize_line(list(percentile_list_train['sent']), vocab)  # Vectorizing train X
test_text_vec, word2index_test = vectorize_line(list(percentile_list_test['sent']), vocab)  # Vectorizing train X_test

ne_vec,ne2index=vectorize_line(list(percentile_list_train['ne']), ne_tags_set)  # Vectorizing train named entities
ne_vec_test,ne2index_test=vectorize_line(list(percentile_list_test['ne']), ne_tags_set)  # Vectorizing test named entities

pos_vec,pos2index=vectorize_line(list(percentile_list_train['pos']), pos_set)  # Vectorizing train POS
pos_vec_test,pos2index_test=vectorize_line(list(percentile_list_test['pos']), pos_set)  # Vectorizing test POS

max_train=0
for i in percentile_list_train["sent"]:
    if(max_train<len(i)):
        max_train=len(i)

train_text_vec = pad_seq(train_text_vec, max_train)  # Padding train X to max_train
test_text_vec = pad_seq(test_text_vec, max_train)  # Padding train X_test to max_test

ne_vec=pad_seq(ne_vec, max_train)
ne_vec_test=pad_seq(ne_vec_test,max_train)

pos_vec=pad_seq(pos_vec, max_train)
pos_vec_test=pad_seq(pos_vec_test,max_train)

tag_set.add('<pad>')
tag_set=list(tag_set)

padded_tags = pad_tags(percentile_list_train['target'], max_train)
tags_vec,tags2index = vectorize_tag(tag_set, padded_tags)  # Vectorizing train y


test_padded_tags = pad_tags(percentile_list_test['target'], max_train)
test_tags_vec,tags2index_test = vectorize_tag(tag_set, test_padded_tags)  # Vectorizing test y



tag2index= {w: i for i, w in enumerate(ne_tags_set)}
verb_vec = [[word2index[word]] for word in percentile_list_train['verbs']]  # Vectorizing Verbs

test_verb_vec = [[word2index_test[word]] for word in percentile_list_test['verbs']]  # Vectorizing Verbs


MAX_LENGTH = len(max(train_text_vec, key=len))  # Finding max length
test_MAX_LENGTH = len(max(test_text_vec, key=len))  # Finding max length


train_sentences_X = train_text_vec  # Just variable transfer

verb_vec_extended=[[word2index[word]]*MAX_LENGTH for word in percentile_list_train['verbs']]  # Vectorizing Verbs
test_verb_vec_extended = [[word2index_test[word]]*test_MAX_LENGTH for word in percentile_list_test['verbs']]  # Vectorizing Verbs

big_x = np.concatenate((train_text_vec, verb_vec), axis=1)  # Concatenating verb and sentences for encoder-decoder model
MAX_LENGTH += 1


big_x_test = np.concatenate((test_text_vec, test_verb_vec), axis=1)  # Concatenating verb and sentences


cat_train_tags_y = to_categorical(tags_vec, len(tag_set))  # Y variable to on hot
cat_test_tags_y = to_categorical(test_tags_vec, len(tag_set))  # Y variable to on hot

#
#
##Encoder Decoder Model
#model = encoder_decoder(MAX_LENGTH, max_train)  # Loading encoder decoder model
#
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
#
##history = model.fit(x=big_x, y=cat_train_tags_y, batch_size=128, epochs=1, verbose=1, validation_split=0.2)  # Train the model
#history = model.fit(x=big_x, y=cat_train_tags_y, batch_size=128, epochs=20, verbose=1, validation_split=0.2,callbacks=[es])  # Train the model

################################## Bi-Directional LSTM #####################################################

embedding_matrix=embed(word2index)
varb_arr=np.array(verb_vec_extended)
ne_arr=np.array(ne_vec)
pos_arr=np.array(pos_vec)

model = Bidirectional_LSTM(MAX_LENGTH-1,embedding_matrix)  # Loading encoder decoder model

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

history = model.fit(x=[train_text_vec,varb_arr,ne_arr,pos_arr], y=cat_train_tags_y, batch_size=128, epochs=120, verbose=1, validation_split=0.2,callbacks=[es])  # Train the model


varb_arr_test=np.array(test_verb_vec_extended)
pos_arr_test=np.array(pos_vec_test)
ne_arr_test=np.array(ne_vec_test)
y_pred=model.predict(x=[test_text_vec,varb_arr_test,ne_arr_test,pos_arr_test])

#############Convert the index into tags and store the output in op.txt####################################### 
result=[]

for i in range(len(y_pred)):
    for j in range(len(percentile_list_test["sent"][i])):
        temp=np.argsort(y_pred[i,j,:])
        if tag_set[temp[len(temp)-1]]!='<pad>':
            result.append(percentile_list_test["sent"][i][j]+" "+tag_set[temp[len(temp)-1]]+" "+percentile_list_test["target"][i][j])

with open('op.txt', 'w') as filehandle:
    for listitem in result:
        filehandle.write('%s\n' % listitem)




