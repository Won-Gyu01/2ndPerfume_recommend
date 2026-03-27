import pandas as pd
import random

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def filtering_mood(datadir, cam):

    pfdf = pd.read_csv(datadir)

    mood = ['anger', 'fear', 'happy', 'neutral', 'sad']
    res = []
    moodv = []
    for i in range(0,len(pfdf)):
        moodv.append(random.choice(mood))

    pfdf['mood'] = moodv

    i = 0
    if(cam != 0):
        genx = pfdf[pfdf.mood.apply(lambda x: mood[cam -1] in x)]
        pf = genx.groupby(['perfume_id']).sum()
    else:
        print("nan이 나와서 아무거나 추천")
        pf = pfdf.groupby(['perfume_id']).sum()
        
    pf = pf.drop(['mood'],axis=1)
    frequent_itemsets = apriori(pf, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets , metric="confidence", min_threshold=0.75)
    # 리프트가 1보다 큰 규칙
    rules = rules[(rules['lift']>1)]
    # 신뢰도의 내림차순으로 정렬
    rules = rules.sort_values(by='confidence',ascending = False)
    rules

    # 신뢰도의 내림차순으로 정렬
    urg = rules.drop_duplicates(['consequents'], keep='first')
    urg = urg.head()
    urule = urg[urg.antecedents.apply(lambda x: mood[cam -1] in x)] # 앞서 구한 장르를 가진 혹은 연관된 장르를 가진 영화들 중 추천 영화를 받고 처음에 검색한 영화와 연관된 영화 찾아서 추출
    urule = urule.drop_duplicates(['consequents'], keep='first')
    urule = urule.head()
    if len(urule) >= 1:
        urule = [list(x)[0] for x in urule['consequents']]
        print("선택하신"+mood[cam -1]+"관련 추천")
        for i in urule:
            print(i)
            res.append(i)
    elif len(urg) >= 1:
        urg = [list(x)[0] for x in urg['consequents']]
        print("선택하신"+mood[cam-1]+"관련 추천")
        for i in urg:
            print(i)
            res.append(i)
    else:
        print("error")

    return res
