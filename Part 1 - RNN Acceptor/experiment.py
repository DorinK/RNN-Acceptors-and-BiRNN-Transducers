import string
import sys
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import random
from gen_examples import generate_language_sequences, sequences_to_file

""""""""""""""""""""""""""
#     Dorin Keshales
""""""""""""""""""""""""""

MODE = str(sys.argv[1])
REGULAR_POS = "generate_positive_seq"
REGULAR_NEG = "generate_negative_seq"
PALINDROME_POS = "generate_palindrome_sequence"
PALINDROME_NEG = "generate_non_palindrome_sequence"
DUPLICATE_WORD_POS = "generate_duplicate_word_sequence"
DUPLICATE_WORD_NEG = "generate_non_duplicate_word_sequence"
CROSS_POS = "generate_cross_sequence"
CROSS_NEG = "generate_cross_sequence"
REGULAR, DUPLICATE_WORD, PALINDROME, CROSS = "regular", "duplicate", "palindrome", "cross"

EPOCHS = 3 if MODE == 'REGULAR' else 13
TRAIN_SIZE = 5000
TEST_SIZE = 750
EMBEDDING_DIM = LSTM_OUTPUT_DIM = 30
HIDDEN_LAYER = 25


class RNNAcceptor(nn.Module):

    def __init__(self, embedding_dim, lstm_output_dim, vocab_size, hidden_layer_size, output_size):
        super(RNNAcceptor, self).__init__()

        self.lstm_output_dim = lstm_output_dim

        # Embedding matrix of chars.
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, self.lstm_output_dim)

        self.linear1 = nn.Linear(self.lstm_output_dim, hidden_layer_size)

        # The linear layer that maps from hidden state space to tag space
        self.linear2 = nn.Linear(hidden_layer_size, output_size)

        # Defining the non linear activation function to be TanH.
        self.activation = nn.ReLU()

    def forward(self, sentence):

        # First take the corresponding embedding vectors to the characters in the sequence.
        embeds = self.embeddings(sentence)

        # Run through the lstm.
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))

        # Activate the ReLU on the linear layer's output.
        x = self.activation(self.linear1(lstm_out[-1].view(1, -1)))

        # For the second linear layer.
        x = self.linear2(x)

        return F.log_softmax(x, dim=-1)


def train(model, optimizer, train_sequences, train_tags, test_sequences, test_tags):

    for epoch in range(EPOCHS):

        # Declaring training mode.
        model.train()

        # Count wall-clock time for each epoch.
        start_time = time()

        # Shuffle the train data
        c = list(zip(train_sequences, train_tags))
        np.random.shuffle(c)
        train_x, train_y = zip(*c)

        sum_loss = 0.0
        for sequence, tag in zip(train_x, train_y):

            # Reset the gradients from the previous iteration.
            model.zero_grad()

            if torch.cuda.is_available():
                sequence = sequence.cuda()
                tag = tag.cuda()

            # Get the input ready for the model.
            sequence = Variable(sequence)

            # Forward pass.
            output = model(sequence)

            # Compute the negative log likelihood loss.
            loss = F.nll_loss(output, tag)
            sum_loss += loss.item()

            # Back propagation- computing the gradients.
            loss.backward()

            # Update the parameters
            optimizer.step()

        # Calculating the loss on the training set in the current epoch.
        train_loss = sum_loss / len(train_sequences)

        # Calculating the accuracy on the training set in the current epoch.
        train_accuracy, tr_loss = accuracy_on_data_set(model, train_sequences, train_tags)

        # Calculating the loss and accuracy on the dev set in the current epoch.
        dev_accuracy, dev_loss = accuracy_on_data_set(model, test_sequences, test_tags)

        passed_time = time() - start_time
        print(epoch, train_loss, tr_loss, train_accuracy, dev_loss, dev_accuracy, passed_time)


def accuracy_on_data_set(model, sequences, tags):

    # Declaring evaluation mode.
    model.eval()

    good = total = 0.0
    sum_loss = 0.0

    with torch.no_grad():

        # Shuffle the data
        c = list(zip(sequences, tags))
        np.random.shuffle(c)
        x, y = zip(*c)

        for sequence, tag in zip(x, y):
            total += 1

            if torch.cuda.is_available():
                sequence = sequence.cuda()
                tag = tag.cuda()

            # Prepare the input to the model.
            sequence = Variable(sequence)

            # Calculating the model's predictions to the examples in the current batch.
            output = model(sequence)

            output = output.detach().cpu()
            tag = tag.cpu()

            # Get the indexes of the max log-probability.
            prediction = np.argmax(output.data.numpy())

            # Calculating the negative log likelihood loss.
            loss = F.nll_loss(output, tag)
            sum_loss += loss.item()

            if prediction == tag:
                good += 1

    # Calculating the loss and accuracy rate on the data set.
    return good / total, sum_loss / len(sequences)


def create_data_set(file_name, func_pos, func_neg, train_size, test_size):

    def create(file, num_sequences):

        # Create the data set.
        positive, negative = generate_language_sequences(func_pos, func_neg, num_sequences)

        # Making the data set supervised.
        positive, negative = {item + ' 1' for item in positive}, {item + ' 0' for item in negative}

        data_set = list(positive | negative)

        # Shuffle the examples in the data set.
        random.shuffle(data_set)

        # Write the data set to a file.
        sequences_to_file(file, data_set)

    create(file_name + "_train", train_size)
    create(file_name + "_test", test_size)


# Reading the data from the requested file.
def read_data(file_name):

    def read(file):
        with open(file, "r", encoding="utf-8") as f:
            sequences, tags = [], []

            data = f.readlines()

        # For each line in the file.
        for line in data:
            sequence, tag = line.strip().split()
            sequences.append(sequence)
            tags.append(tag)

        return sequences, tags

    train_x, train_y = read(file_name + "_train")
    dev_x, dev_y = read(file_name + "_test")

    return train_x, train_y, dev_x, dev_y


# Replace each word in the data set with its corresponding index.
def convert_data_to_indexes(data, c2i):
    sequences = []

    # Go over each sequence in the training set.
    for sequence in data:
        sequence_indexes = [c2i[char] for char in sequence]

        # Keep the words in the data set in sequences order.
        sequences.append(sequence_indexes)

    # Return the updated data
    return sequences


# Replace each tag of a word in the data set with its corresponding index.
def convert_tags_to_indexes(tags, t2i):
    return [t2i[tag] for tag in tags]


def generate_and_read_data_set(file_name, pos_func, neg_func, train_size=TRAIN_SIZE, test_size=TEST_SIZE):
    create_data_set(file_name, pos_func, neg_func, train_size, test_size)
    return read_data(file_name)


if __name__ == '__main__':

    vocab = ""
    prefix = "./"
    train_sequences, train_tags, test_sequences, test_tags = [], [], [], []

    if MODE == REGULAR:

        train_sequences, train_tags, test_sequences, test_tags = generate_and_read_data_set(
            prefix + REGULAR, REGULAR_POS, REGULAR_NEG, 2000, 500)

        vocab = string.digits[1:] + string.ascii_letters[:4]

    elif MODE == PALINDROME:

        train_sequences, train_tags, test_sequences, test_tags = generate_and_read_data_set(
            prefix + PALINDROME, PALINDROME_POS, PALINDROME_NEG)

        vocab = string.digits + string.ascii_letters

    elif MODE == DUPLICATE_WORD:
        train_sequences, train_tags, test_sequences, test_tags = generate_and_read_data_set(
            prefix + DUPLICATE_WORD, DUPLICATE_WORD_POS, DUPLICATE_WORD_NEG)

        vocab = string.digits + string.ascii_letters

    elif MODE == CROSS:

        train_sequences, train_tags, test_sequences, test_tags = generate_and_read_data_set(
            prefix + CROSS, CROSS_POS, CROSS_NEG)

        vocab = string.ascii_lowercase

    else:
        raise ValueError("Wrong representation was entered")

    # Assigning a unique index to each char, tag in the vocabulary.
    char_to_ix = {word: i for i, word in enumerate(sorted(list(vocab)))}
    ix_to_char = {i: word for i, word in enumerate(sorted(list(vocab)))}
    tag_to_ix = {"0": 0, "1": 1}

    # Update to indexes representation.
    train_sequences = convert_data_to_indexes(train_sequences, char_to_ix)
    train_tags = convert_tags_to_indexes(train_tags, tag_to_ix)

    # Make each sequence a tensor.
    train_sequences = [torch.LongTensor(x) for x in train_sequences]
    train_tags = [torch.LongTensor([x]) for x in train_tags]

    # Update to indexes representation.
    test_sequences = convert_data_to_indexes(test_sequences, char_to_ix)
    test_tags = convert_tags_to_indexes(test_tags, tag_to_ix)

    # Make each sequence a tensor.
    test_sequences = [torch.LongTensor(x) for x in test_sequences]
    test_tags = [torch.LongTensor([x]) for x in test_tags]

    # Creating an instance of RNNAcceptor.
    model = RNNAcceptor(embedding_dim=EMBEDDING_DIM, lstm_output_dim=LSTM_OUTPUT_DIM, vocab_size=len(char_to_ix),
                        hidden_layer_size=HIDDEN_LAYER, output_size=len(tag_to_ix))

    if torch.cuda.is_available():
        model.cuda()

    # Using Adam optimizer.
    optimizer = optim.Adam(model.parameters())

    # Training the model.
    train(model, optimizer, train_sequences, train_tags, test_sequences, test_tags)
