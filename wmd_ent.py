import pandas as pd
import MeCab
from gensim.models import KeyedVectors
from gensim.similarities import WmdSimilarity
import numpy as np

# モデル読み込み
def uclid(df, n=5):
    df["wmd"] = [0.0]*len(df)
    return df
    model_path = "../../interview_analysis_demo/entity_vector/entity_vector.model.bin"
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    # CSV読み込み

    # 対話単位、話者単位で処理しやすいようソート
    #df = df.sort_values("segmentID").reset_index(drop=True)

    # 結果格納用リスト
    mecab = MeCab.Tagger("-Owakati")
    # 対象となる対話IDごとにデータを処理
    
    # 対話内で各発話に対してエントレインメントスコアを計算
    # 発話単位でループ
    utterances = df['text'].tolist()
    utterances = map(lambda txt:mecab.parse(txt), utterances)
    #turns = df['segmentID'].tolist()
    speakers = df['speaker'].tolist()

    # ここで、各ターゲット発話について相手話者の直近n発話を取得し、WMDを計算
    score_list = []
    for i, (utt, spk) in enumerate(zip(utterances, speakers)):
        # ターゲット発話と反対の話者を特定
        # 話者が"A"なら"B"、"B"なら"A"といった具合で反転する想定
        # ここは話者ラベルの形式に依存するので適宜調整
        # 例：AとB以外なら、異なる話者の判定方法を別途記述
        
        # 過去の相手発話を抽出
        prev_other_utterances = df[(df['speaker'] != spk) & (df.index < i)]

        # 過去n発話を取得（ない場合もある）
        prev_other_utterances = prev_other_utterances.tail(n)
        if len(prev_other_utterances) == 0:
            # 過去に相手発話がない場合はスコアをNaNまたは特定値とする
            score_list.append(float('nan'))
        else:
            # ターゲット発話と各過去相手発話とのWMDを計算
            # WMD計算のためのトークンリスト化（ここではutteranceが既にトークン化済みと想定）
            target_tokens = utt.strip().split()
            min_distance = float('inf')
            for prev_utt in prev_other_utterances['text']:
                prev_utt = mecab.parse(prev_utt)
                prev_tokens = prev_utt.strip().split()
                # WMD計算
                distance = model.wmdistance(target_tokens, prev_tokens)
                if distance < min_distance:
                    min_distance = distance
            if min_distance == float("inf"):
                min_distance = np.nan
            # 最小距離をスコアとして格納
            score_list.append(min_distance)

    # 対話内の結果をデータフレームに結合
    #df['wmd'] = score_list
    df["wmd"] = 0.0

    
# 最終的なCSV出力
    return df
