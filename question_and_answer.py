import string
from itertools import chain
import remath
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet

def download():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

def preprocessing(sentence):

    # remove punctuations
    exclude = set(string.punctuation)
    sentence = ''.join(ch for ch in sentence if ch not in exclude)

    tokens = nltk.word_tokenize(sentence)

    #remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    #remove prepositions
    tagged = nltk.pos_tag(tokens)
    keywords = [tag[0] for tag in tagged if tag[1] not in ['DT', 'IN']]

    return keywords

def get_synonyms(word):
    # get synonums of word
    synonyms = wordnet.synsets(word)
    lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))

    return lemmas

def apply_synonyms(text, keywords_synonyms):
    words = []
    for word in text.split():
        words.append(word)
        for keyword, synonyms in keywords_synonyms.items():
            if word in synonyms:
                del words[-1]
                words.append(keyword)
                break

    return ' '.join(words)


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def main():
    # remove puctuations, stopwords, and preposition from student's ansuwer and instructor's answer.
    clean_correct_answer = " ".join(preprocessing(correct_answer))
    clean_student_answer = " ".join(preprocessing(student_answer))

    # get keywords from instructor's answer and get synonyms of the keywords as a dict.
    keywords = clean_correct_answer.split()
    keywords_synonyms = dict((word, get_synonyms(word)) for word in keywords)
    # print (clean_correct_answer)
    # print (clean_student_answer)

    # replace the words of the answers in synonyms with keywords
    correct_answer_by_synonyms = apply_synonyms(clean_correct_answer, keywords_synonyms)
    student_answer_by_synonyms = apply_synonyms(clean_student_answer, keywords_synonyms)
    # print (correct_answer_by_synonyms)
    # print (student_answer_by_synonyms)

    # get the vec of the answers
    correct_answer_vec = text_to_vector(correct_answer_by_synonyms)
    student_answer_vec = text_to_vector(student_answer_by_synonyms)
    # print (correct_answer_vec)
    # print (student_answer_vec)

    # Get the cosine similarity between 2 vec
    simililarity = cosine_similarity(correct_answer_vec, student_answer_vec)

    # Get the grade from cosine similarity
    if simililarity > 0.6:
        grade = 'pass'
    else:
        grade = 'fail'

    print (grade)

if __name__ == "__main__":
    correct_answer = "The cost required to develop the system."
    # student_answer = "The charge needed to evolve the system."
    student_answer = "The system needs for development, so we compute the cost to develop it."

    WORD = re.compile(r'\w+')

    download()
    main()
