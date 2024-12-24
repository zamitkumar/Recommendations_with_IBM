Recommendation System Using IBM Watson Studio Data

**Introduction**

In this notebook, we will build a recommendation system using real-world data from the IBM Watson Studio platform. 
The goal is to apply machine learning techniques to recommend products, services, or content based on user preferences, past interactions, or other relevant information.

**Step 1: Import Required Libraries**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors


**Step 2: Load the Data**
**Step 3: Data Preprocessing**

**Part I : Exploratory Data Analysis**
Provide a visual and descriptive statistics to assist with giving a look at the number of times each user interacts with an article.

**Part II: Rank-Based Recommendations**
The popularity of an article can really only be based on how often an article was interacted with.

**Part III: User-User Based Collaborative Filtering**

Use the function below to reformat the df dataframe to be shaped with users as the rows and articles as the columns.
Each user should only appear in each row once.
Each article should only show up in one column.
If a user has interacted with an article, then place a 1 where the user-row meets for that article-column. It does not matter how many times a user has interacted with the article, all entries where a user has interacted with an article should be a 1.
If a user has not interacted with an item, then place a zero where the user-row meets for that article-column.

**Part V: Matrix Factorization**
In this part of the notebook, you will build use matrix factorization to make article recommendations to the users on the IBM Watson Studio platform.
1. You should have already created a user_item matrix above in question 1 of Part III above. This first question here will just require that you run the cells to get things set up for the rest of Part V of the notebook.
   
