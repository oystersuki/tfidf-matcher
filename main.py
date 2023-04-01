import pprint

import MeCab
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# MeCab でテキストを分かち書きする関数
def tokenize(text):
    mecab = MeCab.Tagger("-Owakati")
    parsed = mecab.parse(text)
    return parsed.strip().split()


# テキストデータ
base_text = "【限定配布パッチ適応完全版】ensemble お嬢様シリーズ 全タイトルコンプリートセット【期間限定】"
texts = [
    "ensemble お嬢様シリーズ 全タイトルコンプリートセット",
    "魔法少女まどか☆マギカ フィギュアコレクション　全10種セット",
    "エッチな幼なじみとの夜の過ごし方　同人誌",
    "悪魔城ドラキュラ ゲーム音楽集　CD",
    "クトゥルフ神話TRPGセット",
    "恋愛シミュレーションゲーム「Sweet Heart」",
    "メイドさんと楽しむお掃除フェラサービス　アダルトビデオ",
    "プロ野球チーム「オリックス・バファローズ」公式ユニフォーム",
    "新米看護師さんの癒やしのお仕事　エロゲー",
    "オリジナル同人音楽CD「Fantasia」",
    "ネコ耳メイドのためのお勉強セット",
    "美少女戦士セーラームーン ポスターコレクション　全6種セット",
    "ヴィンテージ時計専門店　アンティークウォッチ",
    "イラストレーター　田中春樹　画集",
    "占い師による結婚運アップ術　書籍",
    "ドキドキ☆メイドカフェ　体験チケット",
    "ホラーゲーム「サイレントヒル」オリジナルサウンドトラック　CD",
    "バンドリ！ガールズバンドパーティ！　ぬいぐるみ　全5種セット",
    "くノ一忍法帖　完全版　DVD-BOX",
    "アメリカンビンテージスタイルのジーンズ　ブランド",
    "ぷよぷよ　フィギュア　全10種セット",
    "セクシー女優　桜井あゆ　写真集",
    "ダンジョンズ＆ドラゴンズ　5th edition　プレイヤーズハンドブック",
    "幻想的な世界観が魅力のイラスト集　「Fantasia」",
    "地下アイドルのライブグッズ　タオル＆Tシャツセット",
    "激辛カレーラーメン　お取り寄せセット",
    "新しい異世界で始める農業生活　ライトノベル",
    "スマートフォン用音楽ゲーム　「Cytus II"
]

# 全てのテキストをリストにまとめる
all_texts = [base_text] + texts

# TfidfVectorizer のインスタンスを作成
vectorizer = TfidfVectorizer(tokenizer=tokenize, use_idf=True, smooth_idf=True)

# テキストデータをベクトル化
tfidf_matrix = vectorizer.fit_transform(all_texts)

# コサイン類似度を計算
cosine_similarities = cosine_similarity(tfidf_matrix)

# ベーステキストと他のテキストの類似度を取得
base_text_similarities = cosine_similarities[0, 1:]

# 結果を表示
print("Base text:", base_text)
print("Similarities to other texts:")

cosine_sim_and_text = []
for i, similarity in enumerate(base_text_similarities):
    cosine_sim_and_text.append({"text": texts[i], "similarity": similarity})
    # print(f"Text {texts[i]}: {similarity:.2f}")

pprint.pprint(sorted(cosine_sim_and_text, key=lambda x: x["similarity"])[::-1])
