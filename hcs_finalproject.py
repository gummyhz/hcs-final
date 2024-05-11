import pandas as pd
import re
import numpy as np
import jieba.posseg as pseg
import networkx as nx
import matplotlib.pyplot as plt
import gensim
import plotly.express as px
from sklearn.decomposition import PCA
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

plt.rcParams['font.sans-serif'] = ['SimHei']
# https://alvinntnu.github.io/python-notes/corpus/jieba.html

# https://stackoverflow.com/questions/34587346/python-check-if-a-string-contains-chinese-character 
cjk_ranges = [
        ( 0x4E00,  0x62FF),
        ( 0x6300,  0x77FF),
        ( 0x7800,  0x8CFF),
        ( 0x8D00,  0x9FCC),
        ( 0x3400,  0x4DB5),
        (0x20000, 0x215FF),
        (0x21600, 0x230FF),
        (0x23100, 0x245FF),
        (0x24600, 0x260FF),
        (0x26100, 0x275FF),
        (0x27600, 0x290FF),
        (0x29100, 0x2A6DF),
        (0x2A700, 0x2B734),
        (0x2B740, 0x2B81D),
        (0x2B820, 0x2CEAF),
        (0x2CEB0, 0x2EBEF),
        (0x2F800, 0x2FA1F)
    ]

def is_cjk(char):
    char = ord(char)
    for bottom, top in cjk_ranges:
        if char >= bottom and char <= top:
            return True
    return False

proteins = ['魚', '牛', '羊', '雞', '豬', '豆', '肉', '鴨']
breads = ['面', '餅', '米', '飯']
etc = ['醤', '酱', '醬', '脯', '鮓', '制', '蔬', '甜', '食', '鹹', '苦', '酸', '辣', '湯', '燕', '水', '參', '汁', '芥', '蘑']

nouns = []
foods = []
recipes = []

# Process the 'Pujiang Wushi Zhongkuilu' (浦江吳氏中饋錄, Song Dynasty Pujiang Woman of the Wu Surname Records on Household Essentials)
with open('books/浦江吳氏中饋錄-MadameWu-1275.txt', 'r', encoding='utf8') as rf:
	mwu = rf.read()
	mwu = mwu[:mwu.find("URN:")]

	# between ◎ and \n there are the titles of recipes
	titles = re.compile('◎([\u4e00-\u9fff]+)\n')
	theseRecipes = re.split(titles, mwu)
	theseRecipes = theseRecipes[1:]
	recipes.extend(theseRecipes)

	words = pseg.cut(mwu)
	theseNouns = [w for w in words if w.flag == 'n']
	nouns.extend(theseNouns)

# Process 'Suiyuan shidan' (隨園食單, Recipes from the Garden of Contentment)
with open('books/隨園食單.txt', 'r', encoding='utf8') as rf:
    ssd = rf.read()
    ssd = ssd[:ssd.find("About this digital edition")]
    ssd = ssd.replace('\u3000', '')
    theseRecipes = ssd.split('\n')
    theseRecipes = [r for r in theseRecipes if r != '']
	
    words = pseg.cut(ssd)
    theseNouns = [w for w in words if w.flag == 'n']
    nouns.extend(theseNouns)
	
# hcs-final\books\山家清供.txt
with open('books/山家清供.txt', 'r', encoding='utf8') as rf:
    ssd = rf.read()
    ssd = ssd[:ssd.find("About this digital edition")]
    ssd = ssd.replace('\u3000', '')
    theseRecipes = ssd.split('\n')
    theseRecipes = [r for r in theseRecipes if r != '']
    print(theseRecipes[:3])

    words = pseg.cut(ssd)
    theseNouns = [w for w in words if w.flag == 'n']
    nouns.extend(theseNouns)

# extract food words
for w in nouns:
	for c in w.word:
		if (c in proteins or (c in breads or c in etc)):
			foods.append(w.word)
			break
foods = set(foods)

# create edges between foods
relations = [[f1, f2, 0] for f1 in foods for f2 in foods if (
	(f1 != f2 and not (len(f1)==1 and f1 in f2)) and not (len(f2)==1 and f2 in f1))]

for p in relations:
	for r in recipes:
		if (p[0] in r and p[1] in r):
			p[2] += 1

relations = [r for r in relations if r[2]!=0]
edges = [tuple(row) for row in relations]

df = pd.DataFrame(relations, columns=['from', 'to', 'recipes'])

#G = nx.Graph()
#G.add_nodes_from(foods)
#G.add_weighted_edges_from(edges)
G=nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.Graph() )
pos = nx.spring_layout(G, k=0.15, iterations=20)
# https://stackoverflow.com/questions/14283341/how-to-increase-node-spacing-for-networkx-spring-layout

nx.draw(G, pos, with_labels=True, node_size=800, node_color="skyblue", font_size=10,  width=df['recipes'])
plt.show()

refined_texts = []
print(recipes[0])
print(recipes[1])
for recipe in recipes[1::2]:
	refined = [char for char in recipe if is_cjk(char)]
	refined_texts.append(refined)
print(refined_texts[0])	


vec_model = gensim.models.Word2Vec(sentences=refined_texts, vector_size=100, window=5, sg=0)
words = []
vecs = []
print("A")
for word in vec_model.wv.index_to_key:
	words.append(word)
	vecs.append(vec_model.wv[word])
print("B")
pca = PCA()
print("C")
my_pca = pca.fit_transform(vecs)
print("D")
data = {"words":words, "pc1":my_pca[:,0], 'pc2':my_pca[:,1]}
df2 = pd.DataFrame(data)
print("E")
fig = px.scatter(df2, x="pc1", y="pc2", text="words", opacity=0)
print("F")
fig.write_html("graph.html")
print("G") 
