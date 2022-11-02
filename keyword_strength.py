import nltk

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from math import log

nltk.download('stopwords')
# nltk.download('punkt')

iso_to_nltk = {
    'eng': 'english',
    'ron': 'romanian',
    'hun': 'hungarian',
    'ara': 'arabic',
    'aze': 'azerbaijani',
    'dan': 'danish',
    'nld': 'dutch',
    'fin': 'finnish',
    'fra': 'french',
    'deu': 'german',
    'ell': 'greek',
    'ind': 'indonesian',
    'ita': 'italian',
    'kaz': 'kazakh',
    'nep': 'nepali',
    'nor': 'norwegian',
    'por': 'portuguese',
    'rus': 'russian',
    'slv': 'slovene',
    'spa': 'spanish',
    'swe': 'swedish',
    'tgk': 'tajik',
    'tur': 'turkish'
}


def preprocess(sentence, language):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in stopwords.words(iso_to_nltk[f'{language}'])]
    return " ".join(filtered_words)


class Keyword:
    def __init__(self, kw, language):
        self.keyword = kw.lower()
        self.processed_keyword = preprocess(kw, language)


class Article:
    def __init__(self, id, title, body, language):
        self.id = id
        self.title = title.lower()
        self.body = body.lower()
        self.processed_title = preprocess(title, language)
        self.processed_body = preprocess(body, language)


class ArticleKeywordResult:
    def __init__(self, art_id, art_title, wc, cc, positions, count, metrics):
        self.art_id = art_id
        self.art_title = art_title
        self.wc = wc
        self.cc = cc
        self.positions = positions
        self.count = count
        self.metrics = metrics

    def __str__(self):
        return f"Article id: {self.art_id}\nTitle: {self.art_title}\nDensity: {self.metrics['density']}\n" \
               f"TF: {self.metrics['tf']}\nTF-IDF: {self.metrics['tf-idf']}"


def get_positions_in_text(text, keyword):
    pos_res = []
    pointer = 0

    while text.find(keyword, pointer) != -1:
        start_index = text.find(keyword, pointer)
        end_index = start_index + len(keyword)
        pos_res.append((start_index, end_index))
        pointer = end_index

    return pos_res


def keywords_solidity(language, data_keywords, data_articles):
    keywords = [Keyword(kw, language) for kw in data_keywords]
    articles = [Article(a['art_id'], a["art_title"], a["art_body"], language) for a in data_articles]

    results = {k.keyword: {"idf": 0, "articles": []} for k in keywords}

    # CALCULATING INITIAL STATS
    for kw in keywords:
        # print(kw)
        for art in articles:
            # BASIC INFO
            # word count
            wc_body = len(art.body.split())
            # character count
            cc = len(art.body)

            # POSITIONS and COUNT
            positions = {
                "title": get_positions_in_text(art.title, kw.keyword),
                "body": get_positions_in_text(art.body, kw.keyword)
            }
            count = {
                "title_count": len(get_positions_in_text(art.processed_title, kw.processed_keyword)),
                "body_count": len(get_positions_in_text(art.processed_body, kw.processed_keyword))
            }

            # METRICS
            # density
            wc_body_processed = len(art.processed_body.split())
            density = count["body_count"] / wc_body_processed * 100
            # tf
            tf = count["body_count"] / wc_body_processed
            metrics = {"density": density, "tf": tf, "tf-idf": 0}

            results[kw.keyword]["articles"].append(
                ArticleKeywordResult(art.id, art.title, wc_body, cc, positions, count, metrics))

    # CALCULATING IDF
    total_nr_of_articles = len(articles)

    for k in results:
        articles_with_kw = 0
        for art in results[k]["articles"]:
            if art.count["body_count"] > 0:
                articles_with_kw += 1
        # idf
        if articles_with_kw == 0:
            idf = 0
        else:
            idf = log(total_nr_of_articles / articles_with_kw)
        results[k]["idf"] = idf

    # CALCULATING TF-IDF
    for kw in results:
        kw_idf = results[kw]["idf"]
        for art_res in results[kw]['articles']:
            art_res.metrics['tf-idf'] = art_res.metrics['tf'] * kw_idf

        # SORTING RESULTS
        results[kw]['articles'].sort(key=lambda x: x.metrics['tf-idf'], reverse=True)

        # CONVERSION
        results[kw]['articles'] = [x.__dict__ for x in results[kw]['articles']]

    return results
