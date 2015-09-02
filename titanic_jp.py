# coding: utf-8

import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor


if __name__ == '__main__':
    # データの入手
    # Data obtained from http://biostat.mc.vanderbilt.edu/DataSets
    data = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.csv')

    # pclassの型を数値から文字列に変換
    data['pclass'] = data['pclass'].map(str)

    # データの確認
    # survived: 1（生存），0（死亡）  
    # pclass: 乗客の社会経済的地位（1:上流，2:中流，3:下流）  
    # name: 氏名  
    # sex: 性別  
    # age: 年齢  
    # sibsp: 同乗したSibling/Spouseの数  
    # parch: 同乗したParent/Childrenの数  
    # ticket: チケットナンバー  
    # fare: 乗船料金  
    # cabin: 船室番号  
    # embarked: 乗船場（C = Cherbourg, Q = Queenstown; S = Southampton）  
    # boat: Lifeboat  
    # body:  Body Identification Number  
    # home.dest:  Home/Destination
    print data.describe

    # 欠損値の確認
    print (len(data) - data.count()) / len(data)

    # 特徴量の選択
    # 欠損値の多い特徴量や，分析に有効でなさそうな特徴量を削除します．  
    # ※本来特徴量の選択は分析の試行錯誤のなかで行うべきですが，演習の都合上最初に行っています．
    data2 = data.drop(['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest'], 1)

    # データの可視化
    # カテゴリデータ（pclass, embarked, sex）は積み上げ棒グラフにして  
    # 変数の割合と，変数内の生死の割合を可視化
    pd.pivot_table(data2, index=['pclass'], columns=['survived'],
                   aggfunc='count')['sex'].plot(kind='barh', stacked=True)

    pd.pivot_table(data2, index=['embarked'], columns=['survived'],
                   aggfunc='count')['pclass'].plot(kind='barh', stacked=True)

    pd.pivot_table(data2, index=['sex'], columns=['survived'],
                   aggfunc='count')['pclass'].plot(kind='barh', stacked=True)

    # 数値データ（age, sibsp, parch, fare）は生／死ごとにヒストグラムで可視化
    h = pd.concat([data2['age'][data2['survived']==0],
                   data2['age'][data2['survived']==1]], axis=1)
    h.columns = ['survived: 0', 'survived: 1']
    h.plot(kind='hist', bins=15, alpha=0.3, color=('r','b'), title='age')

    h = pd.concat([data2['sibsp'][data2['survived']==0],
                   data2['sibsp'][data2['survived']==1]], axis=1)
    h.columns = ['survived: 0', 'survived: 1']
    h.plot(kind='hist', bins=8, alpha=0.3, color=('r','b'), title='sibsp')

    h = pd.concat([data2['parch'][data2['survived']==0],
                   data2['parch'][data2['survived']==1]], axis=1)
    h.columns = ['survived: 0', 'survived: 1']
    h.plot(kind='hist', bins=9, alpha=0.3, color=('r','b'), title='parch')

    h = pd.concat([data2['fare'][data2['survived']==0],
                   data2['fare'][data2['survived']==1]], axis=1)
    h.columns = ['survived: 0', 'survived: 1']
    h.plot(kind='hist', bins=15, alpha=0.3, color=('r','b'), title='fare')

    # 特徴量同士の関係を可視化  
    # カテゴリデータ同士の関係はクロス集計（Cross tabulation）で可視化  
    # ※parch, sibspは数値データですが，取りうる値が限られているためクロス集計を利用
    pd.crosstab(data2['embarked'], data2['pclass'])

    pd.crosstab(data2['embarked'], data2['sex'])

    pd.crosstab(data2['embarked'], data2['sibsp'])

    pd.crosstab(data2['embarked'], data2['parch'])

    pd.crosstab(data2['pclass'], data2['sex'])

    pd.crosstab(data2['pclass'], data2['sibsp'])

    pd.crosstab(data2['pclass'], data2['parch'])

    pd.crosstab(data2['sex'], data2['sibsp'])

    pd.crosstab(data2['sex'], data2['parch'])

    pd.crosstab(data2['sibsp'], data2['parch'])

    # 数値データとカテゴリデータの関係は箱ひげ図（boxplot）で可視化
    data2.boxplot(column='age', by='embarked')

    data2.boxplot(column='age', by='pclass')

    data2.boxplot(column='age', by='sex')

    data2.boxplot(column='age', by='sibsp')

    data2.boxplot(column='age', by='parch')

    data2.boxplot(column='fare', by='embarked')

    data2.boxplot(column='fare', by='pclass')

    data2.boxplot(column='fare', by='sex')

    data2.boxplot(column='fare', by='sibsp')

    data2.boxplot(column='fare', by='parch')

    # 数値データ同士の関係は散布図（Scatter plot）で可視化
    plt.plot(data2['age'], data2['fare'], 'b+')
    plt.xlabel('age')
    plt.ylabel('fare')

    # 欠損値の処理
    # 今回は，数値データは中央値で，カテゴリデータは最頻値で補間します．  
    # ただし，運賃（fare）は社会経済的地位（pclass）と相関があるため※  
    # 等級ごとの中央値で補間します．  
    # ※pclassとfareの可視化結果（boxplot）参照
    age_median = data2['age'].dropna().median()
    embarked_mode = data2['embarked'].dropna().mode().values
    fare_median_c1 = data2['fare'][data2['pclass']=='1'].dropna().median()
    fare_median_c2 = data2['fare'][data2['pclass']=='2'].dropna().median()
    fare_median_c3 = data2['fare'][data2['pclass']=='3'].dropna().median()

    data2.loc[data2['age'].isnull(), 'age'] = age_median
    data2.loc[data2['embarked'].isnull(), 'embarked'] = embarked_mode
    data2.loc[(data2['fare'].isnull()) & (data2['pclass']=='1'), 'fare'] = fare_median_c1
    data2.loc[(data2['fare'].isnull()) & (data2['pclass']=='2'), 'fare'] = fare_median_c2
    data2.loc[(data2['fare'].isnull()) & (data2['pclass']=='3'), 'fare'] = fare_median_c3

    print (len(data2) - data2.count()) / len(data2)

    # カテゴリ変数の処理
    # DictVectorizerはN種類の変数をN個の数値特徴量に変換しますが  
    # これは冗長なので，1つ削除します
    vec = DV()
    data2 = pd.DataFrame(vec.fit_transform(data2.T.to_dict().values()).toarray(),
                         columns=vec.get_feature_names(), index=data2.index)
    del data2['embarked=S']
    del data2['pclass=3']
    del data2['sex=male']

    # scikit-learnの入力に合わせ，特徴量と目的変数を分けます
    data2_y = data2.pop('survived')

    # データの標準化
    # 数値データを平均0，分散1に標準化します．
    standardizer = StandardScaler().fit(data2.loc[:, ['age', 'sibsp', 'parch', 'fare']])
    data2.loc[:, ['age', 'sibsp', 'parch', 'fare']] =    standardizer.transform(data2.loc[:, ['age', 'sibsp', 'parch', 'fare']])

    # モデリング
    # 決定木（Decision tree）を使って生死を予測してみます．  
    # 予測結果は精度（accuracy）で評価します．
    # $$精度 = \frac{正解数}{データ数}$$
    clf0 = DecisionTreeClassifier(random_state=1).fit(data2, data2_y)
    accuracy_score(data2_y, clf0.predict(data2))
    # 0.96562261268143623

    # グリッドサーチと交差検証
    cv = KFold(data2.shape[0], n_folds=10, shuffle=True, random_state=1)
    clf1 = GridSearchCV(DecisionTreeClassifier(random_state=1),
                        [{'max_depth': [3, 5, 7]}],
                        cv=cv, scoring='accuracy', n_jobs=-1).fit(data2, data2_y)
    print clf1.best_params_
    # {'max_depth': 5}
    print clf1.best_score_
    # 0.799083269671505

    # 学習曲線
    cv = KFold(data2.shape[0], n_folds=10, shuffle=True, random_state=1)
    train_sizes, train_scores, test_scores = learning_curve(
            clf1.best_estimator_, data2, data2_y, cv=cv, scoring='accuracy',
            train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure()
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.ylim((0.4, 1.))
    plt.legend(loc="lower right")

    # 学習曲線をみると，  
    # 
    # + 訓練スコアが低い  
    # + 訓練スコアと交差検証スコアの差が小さい  
    # 
    # ことから，**ハイバイアス**な状態であると分かります．
    # 
    # そこで，性能改善のために，  
    # 
    # + 柔軟性の高いモデルに変更する  
    # + 特徴量を追加する  
    # 
    # を検討してみます．

    # モデルの変更
    # モデルをより柔軟性の高い，アンサンブル学習（ブースティング）を利用したGradientBoostingClassifierに変更してみます．
    cv = KFold(data2.shape[0], n_folds=10, shuffle=True, random_state=1)
    clf2 = GridSearchCV(GradientBoostingClassifier(n_estimators=100, random_state=1),
                        [{'max_leaf_nodes': [3, 5, 7, 9],
                          'min_samples_leaf': [9, 13, 17, 21],
                          'learning_rate': [0.3, 0.4, 0.5]}],
                        cv=cv, scoring='accuracy', n_jobs=-1).fit(data2, data2_y)
    print clf2.best_params_
    # {'learning_rate': 0.4, 'max_leaf_nodes': 5, 'min_samples_leaf': 17}
    print clf2.best_score_
    # 0.82352941176470584

    # | グリッドサーチ＋交差検証 | モデル変更
    # | -: |-------------: 
    # |0.799| 0.824

    # 特徴量の追加，変更
    # 欠損率が高く利用を見送っていた**cabin**の情報を利用してみます．  
    # cabinは文字＋数値という形なので，文字と数値に分離して利用します．
    data3 = data2.copy(deep=True)

    data3['cabin'] = data['cabin']
    data3.loc[data3['cabin'].isnull(), 'cabin'] = 'unknown'

    cabin_room = [re.sub('^[^\d]+(\d+).*$', '\\1', c) for c in data3['cabin']]
    data3['cabin_room'] = [int(c) if c.isdigit() else 0 for c in cabin_room]

    data3['cabin'] = [re.sub('^([A-Z]).*$', '\\1', c) for c in data3['cabin']]

    vec = DV()
    data3 = pd.DataFrame(vec.fit_transform(data3.T.to_dict().values()).toarray(),
                         columns=vec.get_feature_names(), index=data3.index)
    del data3['cabin=unknown']

    cv = KFold(data3.shape[0], n_folds=10, shuffle=True, random_state=1)
    clf3 = GridSearchCV(GradientBoostingClassifier(n_estimators=100, random_state=1),
                        [{'max_leaf_nodes': [3, 5, 7, 9],
                          'min_samples_leaf': [5, 9, 13, 17],
                          'learning_rate': [0.1, 0.2, 0.3, 0.4]}],
                        cv=cv, scoring='accuracy', n_jobs=-1).fit(data3, data2_y)

    clf3.best_params_
    # {'learning_rate': 0.3, 'max_leaf_nodes': 5, 'min_samples_leaf': 9}

    clf3.best_score_
    # 0.82887700534759357

    # | グリッドサーチ＋交差検証 | モデル変更 | cabin利用 
    # | -: |-------------: | -: 
    # |0.799| 0.824 | 0.829 

    # 単純に中央値で補間していた年齢の欠損値を，その他の特徴量を使って予測してみます．
    data4 = data3.copy(deep=True)
    title = [re.split('\.|,', n)[1].strip() for n in data['name']]
    data4['immature'] = [1 if t in ['Master', 'Mlle', 'Miss'] else 0 for t in title]
    data4 = data4[['sibsp', 'parch', 'pclass=1', 'pclass=2', 'fare', 'immature']]
    cv = KFold(data4[data['age'].notnull()].shape[0], n_folds=10, shuffle=True,
               random_state=1)

    clf_age = GridSearchCV(DecisionTreeRegressor(random_state=1),
                           [{'max_depth': [2, 3, 4]}],
                           cv=cv,
                           n_jobs=-1).fit(data4[data['age'].notnull()],
                                         data['age'][data['age'].notnull()])
    clf_age.best_params_
    # {'max_depth': 3}

    data3.loc[data['age'].isnull(), 'age'] = clf_age.predict(data4[data['age'].isnull()])

    cv = KFold(data3.shape[0], n_folds=10, shuffle=True, random_state=1)
    clf4 = GridSearchCV(GradientBoostingClassifier(n_estimators=100, random_state=1),
                        [{'max_leaf_nodes': [5, 7, 9, 11],
                          'min_samples_leaf': [13, 17, 21, 25],
                          'learning_rate': [0.1, 0.2, 0.3, 0.4]}],
                        cv=cv, scoring='accuracy', n_jobs=-1).fit(data3, data2_y)

    clf4.best_params_
    # {'learning_rate': 0.2, 'max_leaf_nodes': 7, 'min_samples_leaf': 17}

    clf4.best_score_
    # 0.8304048892284186

    # | グリッドサーチ＋交差検証 | モデル変更 | cabin利用 | age予測 
    # | -: |-------------: | -: | -: 
    # |0.799| 0.824 | 0.829 | 0.830 
