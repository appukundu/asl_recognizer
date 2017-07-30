import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probability = []
    guesses = []
    for ii in range(test_set.num_items):
        best_prob, best_word = None, None
        temp_word_probability = {}
        temp_sequences, temp_lengths = test_set.get_item_Xlengths(ii)
        for word, model in models.items():
            try:
                temp_word_probability[word] = model.score(temp_sequences, temp_lengths)
            except Exception as e:
                temp_word_probability[word] = float("-inf")
            if(best_prob == None or temp_word_probability[word] > best_prob):
                best_prob, best_word = temp_word_probability[word], word
            continue
        probability.append(temp_word_probability)
        guesses.append(best_word)
    return probability, guesses
