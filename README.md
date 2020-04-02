Sequence prediction using encoder-decoder model

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
