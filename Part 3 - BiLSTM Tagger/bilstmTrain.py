import sys
import torch as torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

""""""""""""""""""""""""""
#     Dorin Keshales
""""""""""""""""""""""""""

torch.manual_seed(7)

REP = str(sys.argv[1])
TRAIN_FILE = str(sys.argv[2])
MODEL_FILE = str(sys.argv[3])
DEV_FILE = str(sys.argv[4])
TASK = str(sys.argv[5])

EPOCHS = 5
TRAIN_BATCH_SIZE = 50
SEPARATOR, MAX_WORD_LEN, IS_NER = (' ', 54, False) if TASK == 'pos' else ('\t', 61, True)

if TASK == 'pos':
    DEV_BATCH_SIZE, LR, W_EMBED_DIM, BILSTM_OUTPUT_DIM, CHARS_EMBED_DIM, CHAR_LSTM_OUTPUT_DIM = (
        32, 0.001, 150, 512, 1, 1) if REP == 'a' else (32, 0.001, 1, 512, 15, 50) if REP == 'b' else (
        128, 0.002, 50, 256, 1, 1) if REP == 'c' else (32, 0.001, 50, 512, 15, 50)

else:
    DEV_BATCH_SIZE, LR, W_EMBED_DIM, BILSTM_OUTPUT_DIM, CHARS_EMBED_DIM, CHAR_LSTM_OUTPUT_DIM = (
        32, 0.002, 200, 200, 1, 1) if REP == 'a' else (32, 0.002, 1, 230, 200, 220) if REP == 'b' else (
        32, 0.003, 140, 150, 1, 1) if REP == 'c' else (32, 0.001, 220, 230, 200, 220)


class BiLSTM_Tagger(nn.Module):

    def __init__(self, chars_vocab_size, words_vocab_size, prefix_vocab_size, suffix_vocab_size, tags_vocab_size,
                 words_embedding_dim, bilstm_output_dim, max_word_len, chars_embedding_dim, chars_lstm_output_dim):
        super(BiLSTM_Tagger, self).__init__()

        self.num_layers = 2
        self.chars_vocab_size = chars_vocab_size
        self.words_vocab_size = words_vocab_size
        self.prefix_vocab_size = prefix_vocab_size
        self.suffix_vocab_size = suffix_vocab_size
        self.chars_embedding_dim = chars_embedding_dim
        self.words_embedding_dim = words_embedding_dim
        self.chars_lstm_output_dim = chars_lstm_output_dim
        self.bilstm_output_dim = bilstm_output_dim
        self.max_word_len = max_word_len
        self.output_size = tags_vocab_size - 1

        # Embedding matrices - one for each vocabulary.
        self.char_embeddings = nn.Embedding(self.chars_vocab_size, self.chars_embedding_dim, padding_idx=0)
        self.word_embeddings = nn.Embedding(self.words_vocab_size, self.words_embedding_dim, padding_idx=0)
        self.prefix_embeddings = nn.Embedding(self.prefix_vocab_size, self.words_embedding_dim, padding_idx=0)
        self.suffix_embeddings = nn.Embedding(self.suffix_vocab_size, self.words_embedding_dim, padding_idx=0)

        # Dropout with probability 0.5
        self.dropout = nn.Dropout()

        self.activation = nn.Tanh()

        # Unidirectional lstm which handles chars.
        self.chars_lstm = nn.LSTM(self.chars_embedding_dim, self.chars_lstm_output_dim, batch_first=True)

        # Defining the input dim to the bilstm following the chosen representation.
        self.bilstm_input_dim = self.words_embedding_dim if REP == 'a' or REP == 'c' else self.chars_lstm_output_dim \
            if REP == 'd' else self.chars_lstm_output_dim * self.max_word_len

        # Bidirectional lstm with 2 layers which handles words.
        self.bilstm = nn.LSTM(
            input_size=self.bilstm_input_dim,
            hidden_size=self.bilstm_output_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )

        # Linear layers.
        self.linear1 = nn.Linear(self.bilstm_output_dim * 2, self.output_size)
        self.linear2 = nn.Linear((self.chars_lstm_output_dim * self.max_word_len) + self.words_embedding_dim,
                                 self.chars_lstm_output_dim)

    # Representing the words using the chosen representation.
    def word_representation(self, xW=None, xC=None, prefixes=None, suffixes=None):

        if REP == 'a':

            # Finding the corresponding words embedding vectors.
            return self.word_embeddings(xW)

        elif REP == 'b':

            batch_size, seq_len, ch_len = xC.size()

            # Finding the corresponding chars embedding vectors.
            x = self.char_embeddings(xC)

            x = x.view(-1, 1, self.chars_embedding_dim)

            # Now run through the chars LSTM.
            x, _ = self.chars_lstm(x)

            x = x.view(batch_size, seq_len, -1)

            return x

        elif REP == 'c':

            batch_size, seq_len = xW.size()

            # Find the corresponding embedding vectors of the prefixes and concatenating them into one long vector.
            prefixes = self.prefix_embeddings(prefixes).view(batch_size, -1)

            # Find the corresponding embedding vectors of the words and concatenating them into one long vector.
            x = self.word_embeddings(xW).view(batch_size, -1)

            # Find the corresponding embedding vectors of the suffixes and concatenating them into one long vector.
            suffixes = self.suffix_embeddings(suffixes).view(batch_size, -1)

            x = prefixes + x + suffixes

            x = x.view(batch_size, seq_len, self.words_embedding_dim)

            return x

        elif REP == 'd':
            batch_size, seq_len, ch_len = xC.size()

            # Finding the corresponding chars embedding vectors.
            out_ch = self.char_embeddings(xC)

            out_ch = out_ch.view(-1, 1, self.chars_embedding_dim)

            # Now run through the chars LSTM.
            out_ch, _ = self.chars_lstm(out_ch)

            out_ch = out_ch.view(batch_size, seq_len, -1)

            # Find the corresponding words embedding vectors.
            out_w = self.word_embeddings(xW)

            x = torch.zeros(batch_size, seq_len, out_w.shape[2] + out_ch.shape[2])

            if torch.cuda.is_available():
                x = x.cuda()

            # Making the concatenation between representation a to representation b.
            for i in range(batch_size):
                for idx in range(seq_len):
                    x[i, idx] = torch.tensor(out_w[i, idx].tolist() + out_ch[i, idx].tolist())

            # Fully connected layer.
            x = self.linear2(x)

            x = self.activation(x)

            return x
        else:
            raise ValueError("Wrong representation was entered")

    def forward(self, sentences_length, longest_sentence, xW=None, xC=None, prefixes=None, suffixes=None):

        x = self.word_representation(xW, xC, prefixes, suffixes)

        # Packing the sentences in the batch.
        x = torch.nn.utils.rnn.pack_padded_sequence(x, sentences_length, batch_first=True)

        # Now run through LSTM.
        x, _ = self.bilstm(x)

        # Undo the packing operation.
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=longest_sentence)

        x = x.contiguous().view(-1, x.shape[2])

        # Using dropout to prevent overfitting.
        x = self.dropout(x)

        # Fully connected layer.
        x = self.linear1(x)

        return F.log_softmax(x, dim=-1)

    def predict_test(self, sentences_length, longest_sentence, xW=None, xC=None, prefixes=None, suffixes=None):

        x = self.word_representation(xW, xC, prefixes, suffixes)

        # Packing the sentences in the batch.
        x = torch.nn.utils.rnn.pack_padded_sequence(x, sentences_length, batch_first=True, enforce_sorted=False)

        # Now run through LSTM.
        x, _ = self.bilstm(x)

        # Undo the packing operation.
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=longest_sentence)

        x = x.contiguous().view(-1, x.shape[2])

        # Using dropout to prevent overfitting.
        x = self.dropout(x)

        # Fully connected layer.
        x = self.linear1(x)

        return F.log_softmax(x, dim=-1)


def train(model, optimizer, train_loader, dev_loader, train_lengths, dev_lengths, index_to_tag):

    devAccuracy = []
    best_accuracy = 0.0

    for epoch in range(EPOCHS):

        # Declaring training mode.
        model.train()

        counter = 0

        sum_loss = 0.0
        for batch_idx, (x, x_chars, pref_x, suf_x, y, lengths) in enumerate(train_loader):

            # Counting the sentences in each batch
            counter += x.shape[0]

            # Reset the gradients from the previous iteration.
            model.zero_grad()

            # Forward pass
            outputs, y = forward_passing(model, x, x_chars, pref_x, suf_x, y, lengths, train_lengths)

            # Compute the negative log likelihood loss.
            loss = F.nll_loss(outputs, y.view(-1) - 1, ignore_index=-1)
            sum_loss += loss.item()

            # Back propagation- computing the gradients.
            loss.backward()

            # Update the parameters
            optimizer.step()

            if counter % 500 == 0:

                # Calculating the loss and accuracy on the dev set.
                dev_accuracy, dev_loss = accuracy_on_data_set(model, dev_loader, dev_lengths, index_to_tag)
                # Add to list
                devAccuracy.append(dev_accuracy)

                if dev_accuracy > best_accuracy:
                    best_accuracy = dev_accuracy
                    torch.save(model.state_dict(), MODEL_FILE)

                print("Epoch: {}/{}...".format(epoch + 1, EPOCHS),
                      "Step: {}...".format(counter),
                      "Dev Loss: {:.6f}...".format(dev_loss),
                      "Dev Accuracy: {:.6f}".format(dev_accuracy))
                model.train()

    if best_accuracy == 0:
        torch.save(model.state_dict(), MODEL_FILE)


def accuracy_on_data_set(model, data_set, data_set_lengths, index_to_tag):

    # Declaring evaluation mode.
    model.eval()

    good = total = 0.0
    sum_loss = 0.0

    with torch.no_grad():
        for batch_idx, (x, x_chars, pref_x, suf_x, y, lengths) in enumerate(data_set):

            # Forward pass
            outputs, y = forward_passing(model, x, x_chars, pref_x, suf_x, y, lengths, data_set_lengths)

            outputs = outputs.detach().cpu()
            y = y.cpu()

            # Get the indexes of the max log-probability.
            predictions = np.argmax(outputs.data.numpy(), axis=1)

            # Compute the negative log likelihood loss.
            loss = F.nll_loss(outputs, y.view(-1) - 1, ignore_index=-1)
            sum_loss += loss.item()

            total += sum(lengths).item()

            # For each prediction and tag of an example in the batch
            for y_hat, tag in np.nditer([predictions + 1, y.view(-1).numpy()]):

                # Don't count the padding.
                if tag != 0:

                    if y_hat == tag:

                        # Don't count the cases in which both prediction and tag are 'O' in NER.
                        if IS_NER and index_to_tag[int(y_hat)] == 'O':
                            total -= 1
                        else:
                            good += 1

    # Calculating the loss and accuracy rate on the data set.
    return good / total, sum_loss / sum(data_set_lengths)


def main():

    """" Handling the training set """""

    # Loading the training set.
    train_data, train_prefixes, train_suffixes, train_tags = read_data(TRAIN_FILE)
    train_chars = train_data

    # Consider rare words as unknown words.
    train_data = convert_rare_words_to_unknown_token(train_data)

    # Consider rare prefixes as unknown prefixes.
    train_prefixes = convert_rare_words_to_unknown_token(train_prefixes)

    # Consider rare suffixes as unknown suffixes.
    train_suffixes = convert_rare_words_to_unknown_token(train_suffixes)

    # Assigning a unique index to each char, word, prefix, suffix, tag in the vocabulary.
    w2i, i2w = create_words_vocabulary(train_data)
    c2i, i2c = create_chars_vocabulary(train_chars)
    p2i, i2p = create_words_vocabulary(train_prefixes)
    s2i, i2s = create_words_vocabulary(train_suffixes)
    t2i, i2t = create_tags_vocabulary(train_tags)

    # Update to indexes representation.
    train_data = convert_data_to_indexes(train_data, w2i)
    train_chars = convert_chars_to_indexes(train_chars, c2i)
    train_prefixes = convert_data_to_indexes(train_prefixes, p2i)
    train_suffixes = convert_data_to_indexes(train_suffixes, s2i)
    train_tags = convert_tags_to_indexes(train_tags, t2i)

    # Padding
    train_data, train_data_lengths = pad_sentences(train_data)
    train_chars, _ = pad_sentences_and_words(train_chars)
    train_prefixes, _ = pad_sentences(train_prefixes)
    train_suffixes, _ = pad_sentences(train_suffixes)
    train_tags, _ = pad_sentences(train_tags)

    # Creating a torch loader.
    train_data_set = TensorDataset(torch.LongTensor(train_data), torch.LongTensor(train_chars),
                                   torch.LongTensor(train_prefixes), torch.LongTensor(train_suffixes),
                                   torch.LongTensor(train_tags), torch.LongTensor(np.array(train_data_lengths)))

    train_loader = DataLoader(train_data_set, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

    """"" Handling the dev set """""

    # Loading the dev set.
    dev_data, dev_prefixes, dev_suffixes, dev_tags = read_data(DEV_FILE)
    dev_chars = dev_data

    # Update to indexes representation.
    dev_data = convert_data_to_indexes(dev_data, w2i)
    dev_chars = convert_chars_to_indexes(dev_chars, c2i)
    dev_prefixes = convert_data_to_indexes(dev_prefixes, p2i)
    dev_suffixes = convert_data_to_indexes(dev_suffixes, s2i)
    dev_tags = convert_tags_to_indexes(dev_tags, t2i)

    # Padding
    dev_data, dev_data_lengths = pad_sentences(dev_data)
    dev_chars, _ = pad_sentences_and_words(dev_chars)
    dev_prefixes, _ = pad_sentences(dev_prefixes)
    dev_suffixes, _ = pad_sentences(dev_suffixes)
    dev_tags, _ = pad_sentences(dev_tags)

    # Creating a torch loader.
    dev_data_set = TensorDataset(torch.LongTensor(dev_data), torch.LongTensor(dev_chars),
                                 torch.LongTensor(dev_prefixes), torch.LongTensor(dev_suffixes),
                                 torch.LongTensor(dev_tags), torch.LongTensor(np.array(dev_data_lengths)))

    dev_loader = DataLoader(dev_data_set, batch_size=DEV_BATCH_SIZE, shuffle=False)

    # Creating an instance of BiLSTM.
    model = BiLSTM_Tagger(chars_vocab_size=len(c2i), words_vocab_size=len(w2i), prefix_vocab_size=len(p2i),
                          suffix_vocab_size=len(s2i), tags_vocab_size=len(t2i), words_embedding_dim=W_EMBED_DIM,
                          bilstm_output_dim=BILSTM_OUTPUT_DIM, max_word_len=MAX_WORD_LEN,
                          chars_embedding_dim=CHARS_EMBED_DIM, chars_lstm_output_dim=CHAR_LSTM_OUTPUT_DIM)

    if torch.cuda.is_available():
        model.cuda()

    # Using Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training the model
    train(model, optimizer, train_loader, dev_loader, train_data_lengths, dev_data_lengths, i2t)

    # Saving the dictionaries.
    torch.save({
        'char_to_index': c2i,
        'index_to_char': i2c,
        'word_to_index': w2i,
        'index_to_word': i2w,
        'prefix_to_index': p2i,
        'index_to_prefix': i2p,
        'suffix_to_index': s2i,
        'index_to_suffix': i2s,
        'tag_to_index': t2i,
        'index_to_tag': i2t
    }, './dictFile')


if __name__ == '__main__':
    from bilstmTrain_utils import *
    main()
