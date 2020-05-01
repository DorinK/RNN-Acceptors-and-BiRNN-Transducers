import os
import sys
from torch.utils.data import TensorDataset, DataLoader
from bilstmTrain import BiLSTM_Tagger
from bilstmTrain_utils import convert_data_to_indexes,convert_chars_to_indexes
from bilstmPredict_utils import *

""""""""""""""""""""""""""
#     Dorin Keshales
""""""""""""""""""""""""""

torch.manual_seed(7)

REP = str(sys.argv[1])
MODEL_FILE = str(sys.argv[2])
TEST_FILE = str(sys.argv[3])
DICT_FILE = str(sys.argv[4])
TASK = str(sys.argv[5])

SEPARATOR, MAX_WORD_LEN, IS_NER = (' ', 54, False) if TASK == 'pos' else ('\t', 61, True)

if TASK == 'pos':
    DEV_BATCH_SIZE, LR, W_EMBED_DIM, BILSTM_OUTPUT_DIM, CHARS_EMBED_DIM, CHAR_LSTM_OUTPUT_DIM = (
        32, 0.001, 150, 512, 1, 1) if REP == 'a' else (32, 0.001, 1, 512, 15, 50) if REP == 'b' else (
        128, 0.002, 50, 256, 1, 1) if REP == 'c' else (32, 0.001, 50, 512, 15, 50)

else:
    DEV_BATCH_SIZE, LR, W_EMBED_DIM, BILSTM_OUTPUT_DIM, CHARS_EMBED_DIM, CHAR_LSTM_OUTPUT_DIM = (
        32, 0.002, 200, 200, 1, 1) if REP == 'a' else (32, 0.002, 1, 230, 200, 220) if REP == 'b' else (
        32, 0.003, 140, 150, 1, 1) if REP == 'c' else (32, 0.001, 220, 230, 200, 220)


def test_predictions(model, test_loader, test_lengths, original, index_to_tag):

    sentences = []
    with torch.no_grad():

        for batch_idx, (x, x_chars, pref_x, suf_x, lengths) in enumerate(test_loader):

            # Forward pass
            outputs = forward_passing_test(model, x, x_chars, pref_x, suf_x, lengths, test_lengths)

            outputs = outputs.detach().cpu()

            # Get the index of the max log-probability.
            predictions = np.argmax(outputs.data.numpy(), axis=1)

            predictions += 1
            predictions = predictions.reshape(x.shape[0], max(test_lengths)).tolist()

            # Get predictions of each sentence without counting the padding.
            for sentence, length in zip(predictions, lengths):
                sentences.append(sentence[:length])

    # Clearing the content of the file if it already exists; Otherwise, creating the file.
    if os.path.exists("./test4." + TASK):
        os.remove("./test4." + TASK)
    f = open("./test4." + TASK, "a+")

    for sentence, sentence_preds in zip(original, sentences):

        # For each word in the current original sentence and its corresponding prediction
        for word, prediction in zip(sentence, sentence_preds):

            #  Write to the file.
            f.write("{0} {1}\n".format(word, index_to_tag[prediction]))

        # Add new line after each sentence (following the requested format).
        f.write("\n")

    # Close the file.
    f.close()


def main():

    # Loading all the dictionaries.
    dict = torch.load(DICT_FILE)

    c2i = dict['char_to_index']
    i2c = dict['index_to_char']
    w2i = dict['word_to_index']
    i2w = dict['index_to_word']
    p2i = dict['prefix_to_index']
    i2p = dict['index_to_prefix']
    s2i = dict['suffix_to_index']
    i2s = dict['index_to_suffix']
    t2i = dict['tag_to_index']
    i2t = dict['index_to_tag']

    # Redefining the model.
    model = BiLSTM_Tagger(chars_vocab_size=len(c2i), words_vocab_size=len(w2i), prefix_vocab_size=len(p2i),
                          suffix_vocab_size=len(s2i), tags_vocab_size=len(t2i), words_embedding_dim=W_EMBED_DIM,
                          bilstm_output_dim=BILSTM_OUTPUT_DIM, max_word_len=MAX_WORD_LEN,
                          chars_embedding_dim=CHARS_EMBED_DIM, chars_lstm_output_dim=CHAR_LSTM_OUTPUT_DIM)

    # Loading the state dictionary of the model.
    state_dict = torch.load(MODEL_FILE)
    model.load_state_dict(state_dict)
    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    """ Handling the test set """

    # Loading the test set.
    test_data, test_prefixes, test_suffixes = read_test_data(TEST_FILE)
    original = test_data
    test_chars = test_data

    # Update to indexes representation.
    test_data = convert_data_to_indexes(test_data, w2i)
    test_chars = convert_chars_to_indexes(test_chars, c2i)
    test_prefixes = convert_data_to_indexes(test_prefixes, p2i)
    test_suffixes = convert_data_to_indexes(test_suffixes, s2i)

    # Padding
    test_data, test_data_lengths = pad_test_sentences(test_data)
    test_chars, _ = pad_test_words_and_sentences(test_chars)
    test_prefixes, _ = pad_test_sentences(test_prefixes)
    test_suffixes, _ = pad_test_sentences(test_suffixes)

    # Creating a torch loader.
    test_data_set = TensorDataset(torch.LongTensor(test_data), torch.LongTensor(test_chars),
                                  torch.LongTensor(test_prefixes), torch.LongTensor(test_suffixes),
                                  torch.LongTensor(np.array(test_data_lengths)))

    test_loader = DataLoader(test_data_set, batch_size=32, shuffle=False)

    # Calculating the predictions on the test set and writing it to a file.
    test_predictions(model, test_loader, test_data_lengths, original, i2t)


if __name__ == '__main__':
    main()
