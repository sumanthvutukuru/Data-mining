# store the original dataset in 'df', and create the copy of df, df1 = df.copy()

from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

songd = dict()
userd = dict()
data = np.zeros(shape=(163207, 110001), dtype=int)
User = []
Song = []


def data_initialise():
    with open("./kaggle_visible_evaluation_triplets.txt", "r") as f:
        i = 0
        j = 0
        for line in f:
            user, song, count = line.strip().split('\t')
            if song not in songd:
                songd[song] = i
                Song.append(song)
                i = i+1
            if user not in userd:
                userd[user] = j
                User.append(user)
                j = j+1
        data.reshape(i+1, j+1)
        with open("./kaggle_visible_evaluation_triplets.txt", "r") as f:
            for line in f:
                user, song, count = line.strip().split('\t')
                data[songd[song]][userd[user]] = count


data_initialise()
df = pd.DataFrame(data, columns=User, index=Song)
df1 = df.copy()


def recommend_songs(user, num_recommended_songs):

    print('The list of the songs {} Has Watched \n'.format(user))

    for m in df[df[user] > 0][user].index.tolist():
        print(m)

    print('\n')

    recommended_songs = []

    for m in df[df[user] == 0].index.tolist():

        index_df = df.index.tolist().index(m)
        predicted_rating = df1.iloc[index_df, df1.columns.tolist().index(user)]
        recommended_songs.append((m, predicted_rating))

    sorted_rm = sorted(recommended_songs, key=lambda x: x[1], reverse=True)

    print('The list of the Recommended songs \n')
    rank = 1
    for recommended_song in sorted_rm[:num_recommended_songs]:

        print('{}: {} - predicted rating:{}'.format(rank,
              recommended_song[0], recommended_song[1]))
        rank = rank + 1


def song_recommender(user, num_neighbors, num_recommendation):

    number_neighbors = num_neighbors

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(df.values)
    distances, indices = knn.kneighbors(
        df.values, n_neighbors=number_neighbors)

    user_index = df.columns.tolist().index(user)

    for m, t in list(enumerate(df.index)):
        if df.iloc[m, user_index] == 0:
            sim_songs = indices[m].tolist()
            song_distances = distances[m].tolist()

            if m in sim_songs:
                id_song = sim_songs.index(m)
                sim_songs.remove(m)
                song_distances.pop(id_song)

            else:
                sim_songs = sim_songs[:num_neighbors-1]
                song_distances = song_distances[:num_neighbors-1]

            song_similarity = [1-x for x in song_distances]
            song_similarity_copy = song_similarity.copy()
            nominator = 0

            for s in range(0, len(song_similarity)):
                if df.iloc[sim_songs[s], user_index] == 0:
                    if len(song_similarity_copy) == (number_neighbors - 1):
                        song_similarity_copy.pop(s)

                    else:
                        song_similarity_copy.pop(
                            s-(len(song_similarity)-len(song_similarity_copy)))

                else:
                    nominator = nominator + \
                        song_similarity[s]*df.iloc[sim_songs[s], user_index]

            if len(song_similarity_copy) > 0:
                if sum(song_similarity_copy) > 0:
                    predicted_r = nominator/sum(song_similarity_copy)

                else:
                    predicted_r = 0

            else:
                predicted_r = 0

            df1.iloc[m, user_index] = predicted_r
    recommend_songs(user, num_recommendation)

# for recommendation call song_recommender(user_name,num_neghbour,how_many_song_you_want_to_recommend)
