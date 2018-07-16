from heapq import heappush, nlargest
from textblob import TextBlob


def __get_largest_words(file_name):
    word_value_map = {}
    with open(file_name, 'r', encoding='ISO-8859-1') as file:
        for line in file:
            if not line.startswith(';'):
                for word in line.split():
                    word_value_map[word] = TextBlob(word).sentiment.polarity
    largest_words = nlargest(100, word_value_map.items(), key=lambda i: i[1])
    return [group[0] for group in largest_words]


def get_positive_words():
    return __get_largest_words('sample_words/positive-words.txt')


def get_negative_words():
    return __get_largest_words('sample_words/negative-words.txt')


if __name__ == '__main__':
    print(get_positive_words())
    print(get_negative_words())
