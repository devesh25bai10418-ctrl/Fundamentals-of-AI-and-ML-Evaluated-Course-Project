#Movie
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


movies = pd.DataFrame({
    "title": ["Iron Man","The Dark Knight","Inception","Avengers","Interstellar"],
    "description": [
        "a billionaire builds an armored suit to fight crime.",
        "a masked crusader protects gotham city.",
        "a thief steals information through dream tech.",
        "superheroes team up to stop an alien attack.",
        "explorers travel through a wormhole in space."
    ]
})


vector = TfidfVectorizer(stop_words="english")
tfidf = vector.fit_transform(movies["description"])


similarity = linear_kernel(tfidf, tfidf)

movie_name = "Inception"
movie_index = movies[movies["title"] == movie_name].index[0]


points = similarity[movie_index].argsort()[::-1]


new_list = []
for x in points:
    if x != movie_index:
        new_list.append(x)


result = new_list[:2]

print("Movies like", movie_name)
for r in result:
    print(movies.loc[r, "title"])
