import MeCab
import codecs
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

### データの用意 ###
corpus = [
    "-Mischief Dungeon Life- 異世界転生した俺のイタズラダンジョンライフ",
    "今日はとてもいい天気ですね。",
    "今日は晴れていて、とても暖かい。",
    "明日は雨が降るかもしれません。",
    "今日はとてもいい天気ですね。",
    "今日は晴れていて、とても暖かい。",
    "明日は雨が降るかもしれません。",
    "今日はとてもいい天気ですね。",
    "今日は晴れていて、とても暖かい。",
    "明日は雨が降るかもしれません。",
    "今日はとてもいい天気ですね。",
    "今日は晴れていて、とても暖かい。",
    "明日は雨が降るかもしれません。",
    "今日はとてもいい天気ですね。",
    "今日は晴れていて、とても暖かい。",
    "明日は雨が降るかもしれません。",
]
# corpus[0]
# '原口と槙野も先発だと期待してます。'

### 形態素解析 ###
tagger = MeCab.Tagger('-Owakati')
corpus = [tagger.parse(sentence).strip() for sentence in corpus]
# corpus[0]
# '原口 と 槙野 も 先発 だ と 期待 し て ます 。'

### TF-IDFの計算 ###
vectorizer = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
tfidf = vectorizer.fit_transform(corpus)

### テスト ###
sample = "本田半端ねぇ"
# 分かち書きしたものをリストに入れて渡す
sample_tf = vectorizer.transform([tagger.parse(sample).strip()])
# 本田半端ねぇのTF-IDFを計算する
sample_tfidf = vectorizer.transform(sample_tf)
# コサイン類似度の計算
similarity = cosine_similarity(sample_tfidf, tfidf)[0]

### 結果 ###
# 本田半端ねぇに類似しているツイート上から順に１０個見つける
topn_indices = np.argsort(similarity)[::-1][:10]
for sim, tweet in zip(similarity[topn_indices], np.array(corpus)[topn_indices]):
    print("({:.2f}): {}".format(sim, "".join(tweet.split())))