import numpy as np
import torch
from torch.autograd import Variable
from bilstmPredict import REP, MAX_WORD_LEN

""""""""""""""""""""""""""
#     Dorin Keshales
#       313298424
""""""""""""""""""""""""""


# Reading the data from the requested file.
def read_test_data(file_name):
    sentences, prefixes, suffixes = [], [], []

    with open(file_name, "r", encoding="utf-8") as file:
        data = file.readlines()
        sentence, sentence_prefixes, sentence_suffixes = [], [], []

        # For each line in the file.
        for line in data:

            # As long as the sentence isn't over.
            if line is not '\n':

                word = line.split()[0]
                sentence.append(word)

                # For each word save it's prefix and suffix.
                sentence_prefixes.append(word[:3])
                sentence_suffixes.append(word[-3:])

            else:  # Otherwise.
                sentences.append(sentence)
                prefixes.append(sentence_prefixes)
                suffixes.append(sentence_suffixes)
                sentence, sentence_prefixes, sentence_suffixes = [], [], []

    return sentences, prefixes, suffixes


def pad_test_sentences(sentences):

    # Get the length of each sentence
    lengths = [len(sentence) for sentence in sentences]
    longest_sent = max(lengths)

    # Create an empty matrix with padding tokens
    features = np.zeros((len(sentences), longest_sent), dtype=int)

    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, :len(review)] = np.array(review)

    return features, lengths


def pad_test_words_and_sentences(sentences):

    # Get the length of each sentence
    lengths = [len(sentence) for sentence in sentences]
    longest_sent = max(lengths)
    longest_word = MAX_WORD_LEN

    # Create an empty matrix with padding tokens
    features = np.zeros((len(sentences), longest_sent, longest_word), dtype=int)
    for ii, sentence in enumerate(sentences):
        for idx, word in enumerate(sentence):
            if len(word) != 0:
                features[ii, idx, :len(word)] = np.array(word)
    return features, lengths


def forward_passing_test(model, x_words, x_chars, pref_x, suf_x, lengths, data_set_lengths):

    if REP == 'a':

        if torch.cuda.is_available():
            x_words = x_words.cuda()

        # Get the input ready for the model.
        x_words = Variable(x_words)

        # Forward pass.
        return model.predict_test(lengths, max(data_set_lengths), x_words)

    elif REP == 'b':

        if torch.cuda.is_available():
            x_chars = x_chars.cuda()

        # Get the input ready for the model.
        x_chars = Variable(x_chars)

        # Forward pass.
        return model.predict_test(lengths, max(data_set_lengths), xC=x_chars)

    elif REP == 'c':

        if torch.cuda.is_available():
            x_words = x_words.cuda()
            pref_x = pref_x.cuda()
            suf_x = suf_x.cuda()

        # Get the input ready for the model.
        x_words, pref_x, suf_x = Variable(torch.LongTensor(x_words)), Variable(torch.LongTensor(pref_x)), Variable(
            torch.LongTensor(suf_x))

        # Forward pass.
        return model.predict_test(lengths, max(data_set_lengths), xW=x_words, prefixes=pref_x, suffixes=suf_x)

    elif REP == 'd':

        if torch.cuda.is_available():
            x_words = x_words.cuda()
            x_chars = x_chars.cuda()

        # Get the input ready for the model.
        x_words, x_chars = Variable(x_words), Variable(x_chars)

        # Forward pass.
        return model.predict_test(lengths, max(data_set_lengths), x_words, x_chars)

    else:
        raise ValueError("Wrong representation was entered")
