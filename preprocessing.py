from collections import Counter
import random

def split_file(filename):
    """ Split the content of a file in lines
    This function reads a file as text and splits the
    content by line. Empty lines, including lines with only
    spaces and end of line characters are removed from the result.
    Parameters
    ----------
    filename: str
        The path to the file to split.
    Returns
    -------
    lines: list of str
        The none blank lines from the input file.
    """
    with open(filename, "r", encoding="utf8") as in_file:
        return [line.strip() for line in in_file.readlines() if len(line.strip()) > 0]

def split_line(line):
    """ Split a message line into annotation and message body.
    This function splits a line into an annotation and message body
    using the first occurence of the tab `\t` character. The part
    before the first `\t` in considered as the annotation and the
    part after is the message body.
    Parameters
    ----------
    line: str
        The input string
    Returns
    -------
    annotation: str
        The annotation part of the line
    message: str
        The message part of the line
    Note
    ----
    If `\t` are found in the body of the message, they will be replaced
    by a single space " " character.
    """
    tab_index = line.index("\t")
    ann = line[:tab_index]
    msg = line[tab_index + 1:]
    return ann, msg.replace("\t", "")

def split_lines(lines):
    """ Split a sequence of lines into a sequence of (annotatin, message) tuples
    Parameter
    ---------
    lines: seq of str
        The input lines
    Returns
    -------
    ann_msg_list: list of tuple of str
        A list with `(annotation, message)` tuple for each element of the input sequence.
    """
    return list(map(split_line, lines))

def count_ham_and_spam(ann_msg_list):
    """ Count the number of ham and spam message in the list.
    This functions counts the number of ham and spam messages in
    a list with the same form as the one returned by the `split_lines`
    function.
    Parameters
    ----------
    ann_msg_list: list of tuple of str
        A list containing `(annotation, message)` tuples.
    Returns
    -------
    n_ham: int
        The number of ham messages in the input list
    n_spam: int
        The number of spam messages in the input list
    """
    return (
        len([a for a, _ in ann_msg_list if a == "ham"]),
        len([a for a, _ in ann_msg_list if a == "spam"])
    )

def clean_msg(msg):
    """Clean the content of a message.
    This function cleans the content of the message by replacing
    all non alphabetical characters with white spaces.
    Parameters
    ----------
    msg: str
        The input message
    Returns
    -------
    cleaned_msg: str
        The cleaned message
    """
    return "".join([c if c.isalpha() else " " for c in msg])

def split_words(msg):
    """ Split a message into distinct words.
    This function splits a message into the words composing it.
    It returns the set of distinct words in the message.
    Parameters
    ----------
    msg: str
        The input message.
    Returns
    -------
    words: set of str
        The set of distinct words in the message.
    """
    return set([w.lower() for w in msg.split()])

def count_words(msg_list):
    """Count the number of word occurrences in the corpus.
    This function counts, for each word of in the input corpus (i.e.
    the set of all messages in the list), the number of messages
    containing this word.
    Parameters
    ----------
    msg_list: seq of str
        The list of input messages
    Returns
    -------
    count_dict: dict[str, int]
        The dictionary where keys are the words and values are the counts.
    Note
    ----
    Messages are cleaned by removing non alphabetical characters before splitting words.
    """
    word_dict = Counter()
    for msg in msg_list:
        words = split_words(clean_msg(msg))
        word_dict += Counter(words)
    return word_dict

def count_words_in_ham_and_spam(ann_msg_list):
    """Count the number of ham and spam messages containing each word of the corpus.
    This functions counts the number of ham and spam documents which
    contain at least once a word for each word in the corpus provided
    as the list of messages.
    Parameters
    ----------
    ann_msg_list: list of tuple of str
        The input list of messages given as (annotation, message) tuples.
    Returns
    -------
    ham_spam_count_dict: dict[str, tuple of int]
        The dictionary of counts provided as (word, (n_ham, n_spam)) (key, value) items.
    """
    ham_messages = [m for a, m in ann_msg_list if a == "ham"]
    ham_word_count = count_words(ham_messages)
    spam_messages = [m for a, m in ann_msg_list if a == "spam"]
    spam_word_count = count_words(spam_messages)
    all_words = set(ham_word_count.keys()) | set(spam_word_count.keys())
    ham_spam_word_count = {}
    for w in all_words:
        n_ham = ham_word_count.get(w, 0)
        n_spam = spam_word_count.get(w, 0)
        ham_spam_word_count[w] = (n_ham, n_spam)
    return ham_spam_word_count

def random_split(seq, p=0.5, seed=None):
    """Randomly split a sequence
    Parameters
    ----------
    seq: iterable
        The input sequence
    p: float, optional (default=0.5)
        Probability for an element to be attributed to the first list.
    seed: int, optional (default=None)
        The seed of the random number generator
    Returns
    -------
    list1: list
        The first random list
    list2: list
        The second random list
    """
    if p < 0 or p > 1:
        raise ValueError("The value of 'p' must be a float between 0 and 1 included.")
    list1 = []
    list2 = []
    if seed is not None:
        random.seed(seed)
    for elt in seq:
        if random.random() < p:
            list1.append(elt)
        else:
            list2.append(elt)
    return list1, list2