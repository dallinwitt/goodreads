#import pandas and necessary sklearn modules
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

#create lists of the incomplete book files, incomplete description files, and complete files
incomp_book_files = ['CSV datasets/book1-100k.csv', 
                     'CSV datasets/book100k-200k.csv', 
                     'CSV datasets/book200k-300k.csv', 
                     'CSV datasets/book300k-400k.csv', 
                     'CSV datasets/book400k-500k.csv', 
                     'CSV datasets/book500k-600k.csv']
incomp_desc_files = ['CSV datasets/book1-100k_descrip.csv', 
                     'CSV datasets/book100k-195k_descrip.csv', 
                     'CSV datasets/book195k-295k_descrip.csv', 
                     'CSV datasets/book295k-400k_descrip.csv', 
                     'CSV datasets/book400k-500k_descrip.csv', 
                     'CSV datasets/book500k-600k_descrip.csv']
comp_book_files = ['CSV datasets/book600k-700k.csv',
                   'CSV datasets/book700k-800k.csv',
                   'CSV datasets/book800k-900k.csv', 
                   'CSV datasets/book900k-1000k.csv', 
                   'CSV datasets/book1000k-1100k.csv', 
                   'CSV datasets/book1100k-1200k.csv', 
                   'CSV datasets/book1200k-1300k.csv', 
                   'CSV datasets/book1300k-1400k.csv']

#create empty dfs for each of the file categories
incomplete_book_df = pd.DataFrame(columns = ['Id', 'Name', 'Language'])
incomplete_description_df = pd.DataFrame(columns = ['Id', 'Description'])
comp_book_df = pd.DataFrame(columns = ['Id', 'Name', 'Description', 'Language'])

#read in csv book files for books 1-600k
for path in incomp_book_files:
    df = pd.read_csv(path, usecols = ['Id', 'Name', 'Language'])
    incomplete_book_df = incomplete_book_df.append(df)

#read in csv description files for books 1-600k
for path in incomp_desc_files:
    df = pd.read_csv(path, usecols = ['Id', 'Description'])
    incomplete_description_df = incomplete_description_df.append(df)

#merge incomplete_book_df and incomplete_description_df on 'Id'
book_desc_merge = pd.merge(incomplete_book_df, incomplete_description_df, how = 'inner', on = 'Id')

book_desc_merge.info()

#read in complete files for books 600k-1400k
for path in comp_book_files:
    df = pd.read_csv(path, usecols = ['Id', 'Name', 'Language', 'Description'])
    comp_book_df = comp_book_df.append(df)

comp_book_df.info()

#vertically concatenate the two dfs
books_complete = book_desc_merge.append(comp_book_df)
books_complete.info()

#list unique values in 'Language' column, remove all english and nan values from the list
langs = pd.unique(books_complete['Language']).tolist()
eng_langs = ['eng', 'en-US', 'en-GB', 'enm', 'en-CA', '--']

for lang in eng_langs:
    langs.remove(lang)

langs.pop(0)

#remove all entries in books_complete where Language is in langs
books_eng = books_complete[~books_complete['Language'].isin(langs)]

#drop language col
books_eng.drop('Language', axis = 1, inplace = True)

#use regex to remove html tags
books_eng.replace('<[^>]*>', '', regex = True, inplace = True)
books_eng.dropna(inplace = True)

#take a random sample of 100,000 books from books_eng
books_sample = books_eng.sample(n = 100000, random_state = 42)
books_sample

#convert descriptions to list
desc_list = books_sample['Description'].tolist()
title_list = books_sample['Name'].tolist()

desc_list[:6]

#instantiate TfidfVectorizer as tfidf
tfidf = TfidfVectorizer()

#create a CSR TF-IDF matrix from the description list
csr_mat = tfidf.fit_transform(desc_list)

#create a TruncatedSVD instance, svd, focusing on 1000 words
svd = TruncatedSVD(n_components = 1000)

#create a KMeans instance, kmeans, with 100 clusters
kmeans = KMeans(n_clusters = 100)

#create a pipeline: pipeline
pipeline = make_pipeline(svd, kmeans)

# Fit the pipeline to articles
pipeline.fit(csr_mat)

# Calculate the cluster labels: labels
labels = pipeline.predict(csr_mat)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'title': title_list})

# Display df sorted by cluster label
print(df.sort_values('label'))




