import re
from corpus import corpus
import math
import tabulate


def get_sentences(text):
    return re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+", text)


def get_words(sentence):
    return re.findall(r"\b\w+\b", sentence)


def words_to_lower(words):
    return [word.lower() for word in words]


# Term Frequency Inverse Document Frequency
# https://arxiv.org/pdf/1707.02268
class TFIDF:
    def __init__(self) -> None:
        self.f_D = {}
        self.corpus_length = 0

    def train(self, corpus):
        self.corpus_length = len(corpus)
        print(f"Corpus length: {self.corpus_length}")
        for text in corpus:
            words = words_to_lower(get_words(text))
            words = list(set(words))
            for word in words:
                if word in self.f_D:
                    self.f_D[word] += 1
                else:
                    self.f_D[word] = 1

    def _get_f_d(self, text):
        words = words_to_lower(get_words(text))
        f_d = {}
        for word in words:
            if word in f_d:
                f_d[word] += 1
            else:
                f_d[word] = 1
        return f_d

    def get_tfidf(self, text):
        f_d = self._get_f_d(text)
        tfidf = {}
        for word, f_d_value in f_d.items():
            tf = f_d_value
            f_D_value = self.f_D.get(word, 1)
            idf = math.log(self.corpus_length / f_D_value)
            tfidf[word] = tf * idf
        return sorted(tfidf.items(), key=lambda x: x[0], reverse=True)


class Summarizer:
    def __init__(self, tfidf_scores, tfisf_scores) -> None:
        self.tfidf_scores = tfidf_scores
        self.tfisf_scores = tfisf_scores

    def get_combined_score(self, tfidf_score, tfisf_score):
        max_tfidf = max(self.tfidf_scores, key=lambda x: x[1])[1]
        max_tfisf = max(self.tfisf_scores, key=lambda x: x[1])[1]

        return (tfidf_score / max_tfidf) * (tfisf_score / max_tfisf)

    def summarize(self, text, n):
        sentences = get_sentences(text)
        sentences_scores = []
        for sentence in sentences:
            words = words_to_lower(get_words(sentence))
            score = 0
            for word in words:
                scores = zip(tfidf_scores, tfisf_scores)
                for (tfidf_word, tfidf_score), (tfisf_word, tfisf_score) in scores:
                    if tfidf_word == word and tfisf_word == word:
                        score += self.get_combined_score(tfidf_score, tfisf_score)
            sentences_scores.append((sentence, score))
        sentences_scores = sorted(sentences_scores, key=lambda x: x[1], reverse=True)
        return sentences_scores[:n]


def wrap_text(text, width=80):
    return "\n".join(text[i : i + width] for i in range(0, len(text), width))


if __name__ == "__main__":
    with open("lab1/to_summarize.txt", "r") as f:
        document_to_summarize = f.read()

    tfidf = TFIDF()
    tfidf.train(corpus)

    tfidf_scores = tfidf.get_tfidf(document_to_summarize)

    tfisf = TFIDF()
    tfisf.train(get_sentences(document_to_summarize))

    tfisf_scores = tfisf.get_tfidf(document_to_summarize)

    summarizer = Summarizer(tfidf_scores, tfisf_scores)

    table = []
    headers = ["TFIDF Word", "TFIDF Score", "TFISF", "TFISF Score", "Combined Score"]

    scores = zip(tfidf_scores, tfisf_scores)
    for (tfidf_word, tfidf_score), (tfisf_word, tfisf_score) in scores:
        table.append(
            [
                tfidf_word,
                tfidf_score,
                tfisf_word,
                tfisf_score,
                summarizer.get_combined_score(tfidf_score, tfisf_score),
            ]
        )
    table = sorted(table, key=lambda x: x[4], reverse=True)
    print(tabulate.tabulate(table, headers, tablefmt="grid"))

    summary = summarizer.summarize(document_to_summarize, 3)

    table = []
    headers = ["Sentence", "Score"]

    for sentence, score in summary:
        table.append([wrap_text(sentence), score])

    print(tabulate.tabulate(table, headers, tablefmt="grid"))
