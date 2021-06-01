# Import Libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import Dataset
df = pd.read_csv("E:\\Pycharm Projects\\Recommendation Engine\\movie_dataset.csv")

# Select required features
features = ["keywords", "cast", "genres", "director"]

for feature in features:
    df[feature] = df[feature].fillna("")

# Combine features and create a new column for it
combined_features = df["keywords"]
for col in features[1:]:
    combined_features = combined_features + " " + df[col]
df["combined_features"] = combined_features

# Applying Count Vectorizer
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

# Converting into cosine similarity
cosine_similarity_matrix = cosine_similarity(count_matrix)


# Function to display recommendation
def get_recommendation(top_n):

    # Take user input
    user_entered_movie = input("Please Enter a Movie Name:")

    # Getting index from movie title
    required_index = df[df["title"] == user_entered_movie]["index"].iloc[0]

    # Filtering required row in cosine similarity matrix
    required_cosine_sim_row = cosine_similarity_matrix[required_index]

    # Forming index and similarities pair
    index_similarities = []
    for i in enumerate(required_cosine_sim_row):
        index_similarities.append(i)

    # Sorting index_similarities in descending order based on similarity score
    index_similarity_sorted = sorted(index_similarities, key = lambda x: x[1], reverse = True)

    # Displaying Recommended Movies
    print("Your Recommended Movies are:")
    print("----------------------------------------------------------")
    for index, similarity in index_similarity_sorted[1: top_n+1]:
        required_movie = df[df["index"] == index]["title"].iloc[0]
        print(required_movie)

get_recommendation(5)