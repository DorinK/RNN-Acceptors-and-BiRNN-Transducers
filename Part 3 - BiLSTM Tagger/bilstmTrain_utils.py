import numpy as np
import torch
from torch.autograd import Variable
from collections import Counter
from bilstmTrain import SEPARATOR, MAX_WORD_LEN, REP

""""""""""""""""""""""""""
#     Dorin Keshales
""""""""""""""""""""""""""

# Reading the data from the requested file.
def read_data(file_name):
    sentences, tags, prefixes, suffixes = [], [], [], []

    with open(file_name, "r", encoding="utf-8") as file:
        data = file.readlines()
        sentence, sentence_tags, sentence_prefixes, sentence_suffixes = [], [], [], []

        # For each line in the file.
        for line in data:

            # As long as the sentence isn't over.
            if line is not '\n':

                word, tag = line.strip().split(SEPARATOR)
                sentence.append(word)
                sentence_tags.append(tag)

                # For each word save it's prefix and suffix.
                sentence_prefixes.append(word[:3])
                sentence_suffixes.append(word[-3:])

            else:  # Otherwise.
                sentences.append(sentence)
                prefixes.append(sentence_prefixes)
                suffixes.append(sentence_suffixes)
                tags.append(sentence_tags)
                sentence, sentence_tags, sentence_prefixes, sentence_suffixes = [], [], [], []

    return sentences, prefixes, suffixes, tags


# Considerate rare words like if the were unknown words in order to train the corresponding embedding vector.
def convert_rare_words_to_unknown_token(data, num_occurrences=1, unknown_token='<UNK>'):
    count = Counter()
    convert_to_unk = set()

    # Count the number of occurrences of each word in the training set.
    for sentence in data:
        count.update(sentence)

    # Collect the words in the training set that appear only once.
    for word, amount in count.items():
        if amount <= num_occurrences:
            convert_to_unk.add(word)

    # Go over each sentence in the training set.
    for sentence in data:

        # For each word in the sentence
        for i in range(len(sentence)):

            # If the current word appears only once then considerate it as unknown word.
            if sentence[i] in convert_to_unk:
                sentence[i] = unknown_token

    # Return the updated training set data.
    return data


# Making a words vocabulary where each word has a unique index.
def create_words_vocabulary(data, unknown_token='<UNK>', pad_token='<PAD>'):
    vocab_words = set()

    # Go over each sentence in the data set.
    for sentence in data:

        # For each word in the sentence
        for word in sentence:
            vocab_words.add(word)

    vocab_words.remove(unknown_token)
    vocab_words = sorted(vocab_words)

    # Add the unknown_token.
    vocab_words = [pad_token, unknown_token] + vocab_words

    # Map each word to a unique index.
    word_to_ix = {word: i for i, word in enumerate(vocab_words)}
    ix_to_word = {i: word for i, word in enumerate(vocab_words)}

    return word_to_ix, ix_to_word


# Making a chars vocabulary where each char has a unique index.
def create_chars_vocabulary(data, unknown_token='<UNK>', pad_token='<PAD>'):
    vocab_chars = set()

    # Go over each sentence in the data set.
    for sentence in data:

        # For each word in the sentence
        for word in sentence:

            # For each char in the word
            for ch in word:
                vocab_chars.add(ch)

    vocab_chars = sorted(vocab_chars)

    # Add the unknown_token.
    vocab_chars = [pad_token, unknown_token] + vocab_chars

    # Map each word to a unique index.
    char_to_ix = {char: i for i, char in enumerate(vocab_chars)}
    ix_to_char = {i: char for i, char in enumerate(vocab_chars)}

    return char_to_ix, ix_to_char


# Making a tags vocabulary where each tag has a unique index.
def create_tags_vocabulary(tags, pad_token='<PAD>'):
    vocab_tags = set()

    # Go over each sentence in the data set.
    for sentence_tags in tags:

        # For each tag which belongs to a word in the sentence
        for tag in sentence_tags:
            vocab_tags.add(tag)

    vocab_tags = sorted(vocab_tags)

    vocab_tags = [pad_token] + vocab_tags

    # Map each tag to a unique index.
    tag_to_ix = {tag: i for i, tag in enumerate(vocab_tags)}
    ix_to_tag = {i: tag for i, tag in enumerate(vocab_tags)}

    return tag_to_ix, ix_to_tag


# Replace each word in the data set with its corresponding index.
def convert_data_to_indexes(data, vocab, unknown_token='<UNK>'):
    sentences, words_indexes = [], []

    # Go over each sentence in the training set.
    for sentence in data:

        # For each word in the sentence
        for word in sentence:

            # Find its corresponding index - if not exist then assign the index of the unknown_token.
            ix = vocab.get(word) if word in vocab else vocab.get(unknown_token)
            words_indexes.append(ix)

        # Keep the words in the data set in sentences order.
        sentences.append(words_indexes)
        words_indexes = []

    # Return the updated data
    return sentences


# Replace each char in the data set with its corresponding index.
def convert_chars_to_indexes(data, vocab, unknown_token='<UNK>'):
    sentences, words_indexes, char_indexes = [], [], []

    # Go over each sentence in the training set.
    for sentence in data:

        # For each word in the sentence
        for word in sentence:

            for ch in word:
                # Find its corresponding index - if not exist then assign the index of the unknown_token.
                ix = vocab.get(ch) if ch in vocab else vocab.get(unknown_token)
                char_indexes.append(ix)

            words_indexes.append(char_indexes)
            char_indexes = []

        # Keep the chars in the data set in sentences order.
        sentences.append(words_indexes)
        words_indexes = []

    # Return the updated data
    return sentences


# Replace each tag of a word in the data set with its corresponding index.
def convert_tags_to_indexes(tags, vocab):
    sentences, tags_indexes = [], []

    # Go over each sentence in the training set.
    for sentence in tags:

        # For each tag of a word in the sentence
        for tag_of_word in sentence:

            # Find its corresponding index
            ix = vocab.get(tag_of_word)
            tags_indexes.append(ix)

        # Keep the tags in the data set in sentences order.
        sentences.append(tags_indexes)
        tags_indexes = []

    # Return the updated tags data
    return sentences


def pad_sentences(sentences):
    ordered = sorted(sentences, key=len, reverse=True)

    # Get the length of each sentence
    lengths = [len(sentence) for sentence in ordered]
    longest_sent = max(lengths)

    # Create an empty matrix with padding tokens
    features = np.zeros((len(ordered), longest_sent), dtype=int)

    for ii, review in enumerate(ordered):
        if len(review) != 0:
            features[ii, :len(review)] = np.array(review)

    return features, lengths


def pad_sentences_and_words(sentences):
    ordered = sorted(sentences, key=len, reverse=True)

    # Get the length of each sentence
    lengths = [len(sentence) for sentence in ordered]
    longest_sent = max(lengths)
    longest_word = MAX_WORD_LEN

    # Create an empty matrix with padding tokens
    features = np.zeros((len(ordered), longest_sent, longest_word), dtype=int)

    for ii, sentence in enumerate(ordered):
        for idx, word in enumerate(sentence):
            if len(word) != 0:
                features[ii, idx, :len(word)] = np.array(word)

    return features, lengths


# Preparing the data for the forward pass
def forward_passing(model, x_words, x_chars, pref_x, suf_x, y, lengths, data_set_lengths):

    if REP == 'a':
        if torch.cuda.is_available():
            x_words = x_words.cuda()
            y = y.cuda()

        # Get the input ready for the model.
        x_words = Variable(x_words)

        # Forward pass.
        return model(lengths, max(data_set_lengths), x_words), y

    elif REP == 'b':

        if torch.cuda.is_available():
            x_chars = x_chars.cuda()
            y = y.cuda()

        # Get the input ready for the model.
        x_chars = Variable(x_chars)

        # Forward pass.
        return model(lengths, max(data_set_lengths), xC=x_chars), y

    elif REP == 'c':

        if torch.cuda.is_available():
            x_words = x_words.cuda()
            pref_x = pref_x.cuda()
            suf_x = suf_x.cuda()
            y = y.cuda()

        # Get the input ready for the model.
        x_words, pref_x, suf_x = Variable(torch.LongTensor(x_words)), Variable(torch.LongTensor(pref_x)), Variable(
            torch.LongTensor(suf_x))

        # Forward pass.
        return model(lengths, max(data_set_lengths), xW=x_words, prefixes=pref_x, suffixes=suf_x), y

    elif REP == 'd':

        if torch.cuda.is_available():
            x_words = x_words.cuda()
            x_chars = x_chars.cuda()
            y = y.cuda()

        # Get the input ready for the model.
        x_words, x_chars = Variable(x_words), Variable(x_chars)

        # Forward pass.
        return model(lengths, max(data_set_lengths), x_words, x_chars), y

    else:
        raise ValueError("Wrong representation was entered")
