from nltk import sent_tokenize, word_tokenize, SnowballStemmer
from nltk.corpus import stopwords


TEXT = """
La acțiune au participat polițiștii Serviciului pentru Acțiuni Speciale, împreună cu lucratori din cadrul Inspectoratului pentru Situații de Urgență și cu sprijinul jandarmilor. Aceștia au intervenit în forță și au forțat ușa de acces în locuință, pentru a pătrunde în interior.Bărbatul a fost imobilizat și ulterior extras din locuință, în vederea transportării la o unitate medicală. Acțiunea în forță a avut loc după ce soția bărbatului reclamase anterior la 112 că a fost sechestrată în locuinţă de individul cunoscut cu probleme psihice. La faţa locului s-au deplasat mai multe echipaje, dar şi un negociator, din primele date stabilindu-se că bărbatul nu este agresiv.
"""

# LANGUAGE for both STOPWORDS and STEMMER --> print(stopwords.fileids()); --> nltk.download()

# available languages: arabic, danish, dutch, english, finnish, french, german, hungarian, italian,
# norwegian, portuguese, romanian, russian, spanish, swedish

LANGUAGE = "romanian"

# PERCENTAGE OF THE SUMMARY
PERCENTAGE = 10

QUOTE_MARKS = [
    ("’", "‘"), ("「", "」"), ("»", "«"), ("«", "»"), ("《", "》"), ("„", "”"), ("”", "”"), ("’", "’"),
    ("\"", "\""), ("„", "“"), ("“", "”"), ("‘", "’"), ("『", "』")
]


def find_start_mark(char):
    for i in QUOTE_MARKS:
        if char == i[0]:
            return i[0]


def find_end_mark(char):
    for i in QUOTE_MARKS:
        if char == i[1]:
            return i[1]


def frequency_matrix_per_sentence(sentences, language):
    frequency_matrix = {}
    stop_words = set(stopwords.words(language))
    ps = SnowballStemmer(language=language)

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stop_words:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


def sentences_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    punctuation = ["!", "(", ")", "-", "[", "]", "{", "}", ";", ":", "", "\"", "\\", ",", "<", ">", ".", "/", "?",
                   "@", "#", "$", "%", "^", "&", "*", "_", "~", "``", "", "”", "“"]
    sorted_word_per_doc_table = {k: v for k, v in
                                 sorted(word_per_doc_table.items(), key=lambda item: item[1], reverse=True) if
                                 k not in punctuation}

    return list(sorted_word_per_doc_table.keys())[0:5]


def summarize(text, language, percentage=10):
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)

    frequency_matrix = frequency_matrix_per_sentence(sentences, language)
    count_doc_per_words = sentences_per_words(frequency_matrix)
    frequency_matrix = list(frequency_matrix.items())

    # SENTENCE SCORES
    sentence_values = {}
    for k, freq_table in frequency_matrix:
        score = 0
        for w, c in list(freq_table.items()):
            if w in count_doc_per_words:
                score += 1 * c
        sentence_values[k] = score

    # AVERAGE SCORE
    summation = 0
    for sent, value in list(sentence_values.items()):
        summation += value
    average_score = summation / total_sentences

    # SUMMARY GENERATION
    summary = []
    modifier = 5
    # percentage = abs(percentage) if abs(percentage) <= 100 else 10  # calibrate if given percentage > 100 or < 0
    summary_percentage_from_total = 0

    while summary_percentage_from_total <= percentage:
        # reset summary
        summary = []
        for sentence in sentences:
            if sentence[:15] in sentence_values and sentence_values[sentence[:15]] >= (modifier * average_score):
                summary.append(sentence)
                summary_percentage_from_total = (len(summary) / total_sentences) * 100
                # if summary_percentage_from_total >= percentage:  # check if percentage is reached (!)
                #     break

        modifier -= 0.05

    output = " ".join(summary)

    # QUOTES

    quotes = []
    inside_quote = False
    quote = []
    pair = []
    start_index = 0
    end_index = 0

    for char in enumerate(text):
        if char[1] == find_start_mark(char[1]) and not inside_quote:
            mark_start = find_start_mark(char[1])
            pair.append(mark_start)
            start_index = char[0]
            quote.append(start_index)
            inside_quote = True
            continue

        if char[1] == find_end_mark(char[1]):
            mark_end = find_end_mark(char[1])
            pair.append(mark_end)
            end_index = char[0]
            quote.append(end_index + 1)
            quotes.append((quote[0], quote[1], pair[0], pair[1]))
            inside_quote = False
            quote = []
            pair = []

    # MARKING THE QUOTES in the summary

    result = []

    for sentence in sent_tokenize(output):
        replacement = sentence
        sentence_start = text.find(sentence)
        sentence_end = sentence_start + len(sentence)

        for quote in quotes:
            if sentence_start in range(quote[0], quote[1]) or sentence_end in range(quote[0], quote[1]):

                if sentence_start > quote[0]:
                    replacement = quote[2] + replacement

                if sentence_end < quote[1]:
                    replacement += quote[3]

        result.append(replacement)

    result = " ".join(result)

    return result


print(summarize(TEXT, LANGUAGE, PERCENTAGE))
