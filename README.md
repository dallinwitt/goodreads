# Clustering Books Based on Their Goodreads Summary
## Unsupervised Machine Learning 

#### Motivation:
The goal of this project was to create an unsupervised machine learning algorithm that could cluster books together with other books of a similar topic. Categorization tasks are frustrating and time-consuming to do by hand, and automating a large part of it, even if the results are imperfect, would make the task significantly easier and faster.

#### Methods:
To accomplish this task, I will take all of the book summaries from Goodreads and compile them into a TF-IDF (term frequency / inverse document frequency) matrix. The ML model will cluster the books using a k-means method. The number of clusters, k, used will be arbitrary, but could easily be adjusted based on the needs of a given end-user.

#### Outcomes:
Unlike a supervised ML algorithm, there is not a quantitative way to measure the real-world effectiveness of this clustering. However, examining the clusters seems to reveal an effective and logical categorization of the books. 

A system like this could be scaled up to help build out a recommendation algorithm, or to partially automate categorization of new titles.
