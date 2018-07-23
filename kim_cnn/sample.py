from heapq import heappush, nlargest
from textblob import TextBlob


def __get_largest_words(file_name):
    word_value_map = {}
    with open(file_name, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            if not line.startswith(';'):
                for word in line.split():
                    word_value_map[word] = TextBlob(word).sentiment.polarity
    largest_words = nlargest(150, word_value_map.items(), key=lambda i: i[1])
    return [group[0] for group in largest_words]


def __get_positive_words():
    return __get_largest_words('sample_words/positive-words.txt')


def __get_negative_words():
    return __get_largest_words('sample_words/negative-words.txt')


# return a 100*3*300 vector
def get_words_vector(dim):
    # for testing: use 50 positive words + 50 negative words to init the vector
    word_list_positive = __get_positive_words()
    word_list_negative = __get_negative_words()
    all_words_vector_positive = []
    all_words_vector_negative = []
    with open('../../GoogleNews-vectors-negative300.txt', 'r', encoding='ISO-8859-1') as file:
        for line in file:
            if len(all_words_vector_positive) < 50 and line.split()[0] in word_list_positive:
                word_single_array = [float(num) for num in line.split()[1:]]
                word_array_with_dim = [word_single_array, ]*dim
                all_words_vector_positive.append(word_array_with_dim)
            elif len(all_words_vector_negative) < 50 and line.split()[0] in word_list_negative:
                word_single_array = [float(num) for num in line.split()[1:]]
                word_array_with_dim = [word_single_array, ] * dim
                all_words_vector_negative.append(word_array_with_dim)
    all_words_vector = all_words_vector_positive + all_words_vector_negative
    return all_words_vector


if __name__ == '__main__':
    print(get_words_vector(3))
