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
def get_words_vector():
    # for testing: only use negative words (100 words)
    word_list = __get_negative_words()
    all_words_vector = []
    with open('../../GoogleNews-vectors-negative300.txt', 'r', encoding='ISO-8859-1') as file:
        for line in file:
            if len(all_words_vector) < 100 and line.split()[0] in word_list:
                word_single_array = [float(num) for num in line.split()[1:]]
                word_array_with_three_dim = [word_single_array, ]*3
                all_words_vector.append(word_array_with_three_dim)
    return all_words_vector


if __name__ == '__main__':
    print(get_words_vector())
