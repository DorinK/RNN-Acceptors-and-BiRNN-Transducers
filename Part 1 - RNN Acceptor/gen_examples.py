import string
import rstr
import os

""""""""""""""""""""""""""
#     Dorin Keshales
""""""""""""""""""""""""""

MAX_LEN = 15
NUM_SEQUENCES = 500
STRING_LEN = 50
SEQUENCE_LEN = 100


# Generating random number (1-9) up to 15 digits long.
def generate_random_number():
    return rstr.rstr(string.digits[1:], 1, MAX_LEN)


# Generating sequence of the same letter up to 'length' long.
def generate_letter(ch, length=MAX_LEN):
    return rstr.rstr(ch, 1, length)


# Generating random letter (a-z).
def generate_random_letter():
    return rstr.rstr(string.ascii_lowercase, 1)


# Generating random string ([0-9][a-z][A-Z) up to 'num_characters' characters long.
def generate_random_string(num_characters):
    return rstr.rstr(string.digits + string.ascii_letters, 1, num_characters)


# Generating positive sequence of the regular language.
def generate_positive_seq():
    return "" + generate_random_number() + generate_letter('a') + generate_random_number() + generate_letter(
        'b') + generate_random_number() + generate_letter('c') + generate_random_number() + generate_letter(
        'd') + generate_random_number()


# Generating negative sequence of the regular language.
def generate_negative_seq():
    return "" + generate_random_number() + generate_letter('a') + generate_random_number() + generate_letter(
        'c') + generate_random_number() + generate_letter('b') + generate_random_number() + generate_letter(
        'd') + generate_random_number()


# Writing the generated sequences to a file.
def sequences_to_file(file_name, sequences):

    # Remove the file if already exists.
    if os.path.exists(file_name):
        os.remove(file_name)

    with open(file_name, 'a+', encoding='utf-8') as file:
        for item in sequences:
            file.write("{0}\n".format(item))


# Generate positive and negative sequences.
def generate_sequences(func_pos, func_neg, num=NUM_SEQUENCES):
    pos_sequences, neg_sequences = set(), set()

    # Generate positive sequences.
    while len(pos_sequences) != num:
        pos_sequences.add(eval(func_pos)())

    # Generate negative sequences.
    while len(neg_sequences) != num:
        neg_sequences.add(eval(func_neg)())

    return pos_sequences, neg_sequences


# Generate positive sequences of the "Palindrome" language.
def generate_palindrome_sequence():
    random_string = generate_random_string(STRING_LEN)
    random_char = rstr.rstr(string.digits + string.ascii_letters, 0, 1)
    return random_string + random_char + random_string[::-1]


# Generate negative sequences of the "Palindrome" language.
def generate_non_palindrome_sequence():
    return generate_random_string(101)


# Generate positive sequences of the "Duplicate Word" language.
def generate_duplicate_word_sequence():
    return generate_random_string(STRING_LEN) * 2


# Generate negative sequences of the "Duplicate Word" language.
def generate_non_duplicate_word_sequence():
    return generate_random_string(SEQUENCE_LEN)


# Generate positive sequences of the "Cross" language.
def generate_cross_sequence():
    letters = set()

    # Generate for different random letters.
    while len(letters) != 4:
        letters.add(generate_random_letter())

    first = generate_letter(letters.pop(), 25)
    second = generate_letter(letters.pop(), 25)
    third = letters.pop() * len(first)
    forth = letters.pop() * len(second)
    return first + second + third + forth


# Generate negative sequences of the "Cross" language.
def generate_non_cross_sequence():
    return rstr.rstr(string.ascii_lowercase, 1, SEQUENCE_LEN)


# Generate positive and negative sequences of a language.
def generate_language_sequences(func_pos, func_neg, num=NUM_SEQUENCES):
    pos_sequences, neg_sequences = set(), set()

    # Generate positive sequences.
    while len(pos_sequences) != num / 2:
        pos_sequences.add(eval(func_pos)())

    # Generate negative sequences.
    while len(neg_sequences) != num / 2:
        sequence = eval(func_neg)()
        if sequence not in pos_sequences:
            neg_sequences.add(sequence)

    return pos_sequences, neg_sequences


if __name__ == '__main__':

    positive, negative = generate_sequences("generate_positive_seq", "generate_negative_seq")
    sequences_to_file("./pos_examples", positive)
    sequences_to_file("./neg_examples", negative)
