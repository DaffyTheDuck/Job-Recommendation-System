import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class JobRecommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df.dropna(subset=['title', 'description'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df['text'] = self.df['title'] + " " + self.df['description'].fillna("") + " " + self.df['skills_desc'].fillna("")
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['text'])

    def recommend(self, user_input, top_n=5, location_filter=None, type_filter=None):
        input_vec = self.vectorizer.transform([user_input])
        cosine_similarities = cosine_similarity(input_vec, self.tfidf_matrix).flatten()
        self.df['similarity_score'] = cosine_similarities

        filtered_df = self.df.copy()
        if location_filter:
            filtered_df = filtered_df[filtered_df['location'].fillna("").str.contains(location_filter, case=False, na=False)]
        if type_filter:
            filtered_df = filtered_df[filtered_df['formatted_work_type'].fillna("").str.contains(type_filter, case=False, na=False)]

        top_indices = filtered_df.sort_values(by='similarity_score', ascending=False).head(top_n).index
        return self.df.loc[top_indices][['title', 'company_name', 'location', 'formatted_work_type', 'description', 'job_posting_url', 'similarity_score']]

    def recommend_from_resume(self, resume_text, top_n=5):
        return self.recommend(resume_text, top_n=top_n)
