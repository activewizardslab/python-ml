
# coding: utf-8

### Movielens

##### 1. Loading modules.

# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##### 2. Reading data.

# In[4]:

# we are using pandas "read_table" to extract data from raw files
movies=pd.read_table('movies.dat.txt',delimiter='::', header=None, names=["MovieID", "Title", "Genres"])
ratings=pd.read_table('ratings.dat.txt',delimiter='::', header=None, names=["UserID", "MovieID", "Rating", "Timestamp"])
users=pd.read_table('users.dat.txt',delimiter='::', header=None, names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])


##### 3. Preparing data.

# In[5]:

#Using pandas’s merge function, we first merge ratings with users then merging that result with the movies data. 
data = pd.merge(pd.merge(ratings, users), movies)
data.head(n=5)


##### 4. Call describe for some data.

# In[6]:

data[['Rating', 'Age']].describe()


# What we see:
# 
# - people, who watches movies, are less then 56 years old;
# 
# - most common age is less then 35;
# 
# - average age is about 30 years;
# 
# - average rating is more than 3.5 stars.

##### 5. Most common ratings.

# In[81]:

get_ipython().magic(u'pylab inline')
# we are using matplotlib library to plot histogram, which show most common ratings in this dataset. 
plt.hist(data.Rating, bins=10, color='blue', alpha=0.8)
plt.xlabel("Stars")


# We can see:
# 
# - people like a big part of movies and evaluate them from 3 to 5 stars.

##### 6. Find mean movie ratings for each film grouped by gender.

# In[60]:

# we are using DataFrame's pivot_table method that allow group data by column very easy. 
mean_ratings = data.pivot_table('Rating', rows='Genres', cols='Gender', aggfunc = 'mean')


# In[61]:

# we want to find the genres of movies by gender. 
# we can add a column to mean_ratings containing the difference in means, then sort by that.
mean_ratings['diff'] = mean_ratings['F'] - mean_ratings['M']
sorted_by_diff = mean_ratings.sort_index(by='diff')
sorted_by_diff[:10]


# What we see:
# 
# - top 10 genres of movies that men find better.

##### 7. Do really only women watch movies of the genre Romance?

# In[12]:

# count all genres of movies and group by gender
count_romance = data.pivot_table('Rating', rows='Genres', cols='Gender', aggfunc = 'count')


# In[40]:

count_romance[:10]


# In[51]:

# count number of women and men who watch Romance genre.
num_female_romance=0
num_male_romance=0

for i in range(len(count_romance.index)):
    if (count_romance.index[i].startswith("Romance")) == True:
        num_female_romance += count_romance.F[i]
        num_male_romance += count_romance.M[i]
print num_female_romance, num_male_romance 


# In[52]:

# count general number all women and men in dataset 
num_female = count_romance.F.sum()
num_male = count_romance.M.sum()
num_female, num_male


# In[53]:

# find percentage of the people who watch the movies to all viewers by gender
f = 100 * float(num_female_romance)/float(num_female) 
m = 100 * float(num_male_romance)/float(num_male)
f, m


# We can see that the number of women who watch romances is more than double in comparison with the number of men.

# In[76]:

# find position of Romance genre
for i in range(len(mean_ratings.index)):
    if (mean_ratings.index[i].startswith("Romance")) == True:
        print i    


# In[58]:

# how women and men evaluate movies in genre Romance? The column "diff" show difference between rating each gender.  
romance_mean  = mean_ratings[290:294]
romance_mean.index = range(len(romance_mean))
print romance_mean


# In[78]:

# find mean of rating for Romance
romance_mean.F.mean(), romance_mean.M.mean()


# In[79]:

# find difference between means
diff = romance_mean.F.mean() - romance_mean.M.mean()
diff


# Women give a romance a bit higher rating than men.
# 

##### 8. Which movies do men and women most disagree on?

# In[40]:

# we have DataFrame with multi-level index,  therefore we must return new DataFrame with labeling information in the columns under the index names
data.reset_index('MovieID', inplace=True)


# In[41]:

# we want to find the movies that are most divisive between male and female.
data1 = data.pivot_table(rows=['MovieID','Title'], cols=['Gender'], values='Rating', fill_value=0)
print data1.head()


# In[42]:

# 'diff' gives us the movies with rating difference and which were preferred by women
data1['diff'] = data1.M - data1.F
print data1.head()


# In[43]:

# choose top 50 movies 
most_50 = data.groupby('MovieID').size().order(ascending=False)[:50]
data1.reset_index('MovieID', inplace=True)


# In[44]:

get_ipython().magic(u'matplotlib inline')

disagreements = data1[data1.MovieID.isin(most_50.index)]['diff']
disagreements.order().plot(kind='barh', figsize=[9, 15])
plt.title('Male vs. Female Avg. Ratings\n(Difference > 0 = Favored by Men)')
plt.ylabel('Title')
plt.xlabel('Average Rating Difference');


##### 9. Which occupation watches which Genre the most?

# In[47]:

# values of occupation we found here: https://github.com/pydata/pydata-book/tree/master/ch02/movielens
occupation = ["Genres", "other", "academic/educator", "artist", "clerical/admin", "college/grad student", 
              "customer service", "doctor/health care", "executive/managerial", "farmer", "homemaker",
              "K-12 student", "lawyer", "programmer", "retired", "sales/marketing", "scientist", "self-employed",
              "technician/engineer", "tradesman/craftsman", "unemployed", "writer"]


# In[76]:

# get data with occupation values group by Genres
genres_by_occupation = data.pivot_table('Rating', rows='Genres', cols='Occupation', aggfunc = 'count')


# In[77]:

# next functions allow rename values of genres by first category.
def group_genres(df):
    for i,item in enumerate(df.index):
        word = item.partition('|')
        df['Genres1'] = word[0]
    for i,item in enumerate(df.index):
        word = item.partition('|')
        df['Genres1'][i] = word[0]


# In[78]:

# call a function
group_genres(genres_by_occupation)


# In[81]:

# group by genres
genres_by_occupation = genres_by_occupation.groupby("Genres1", as_index=False)
genres_by_occupation = genres_by_occupation.aggregate(sum)
genres_by_occupation


# In[85]:

# sort by "other" occupation to see with movies they prefer
sorted_by_other = genres_by_occupation.sort_index(by=0)
sorted_by_other.index = range(len(sorted_by_other))
# give the columns right names by occupations
sorted_by_other.columns = occupation
sorted_by_other


# What we see:
# 
# - almost all occupations prefer comedy, action or drama;
# 
# - the least popular are Fantasy and War.

# In[89]:

# let's choose 10 occupation and build plot by number of ratings for each genre
occupation1 = occupation[1:10]
df1 = sorted_by_other[occupation1]
# we are plotting with help of pandas method "plot()"
df1.plot(kind='barh', figsize=[25, 10], x=sorted_by_other.Genres, stacked=True);


####### # Let's see what films watch writers?

# In[92]:

df = sorted_by_other[['Genres', 'writer']]


# In[93]:

df.plot(kind='line', x=df.Genres, figsize=[9, 3])
plt.title('What movies watch writers?')
plt.ylabel('count')
plt.xlabel('genres');


##### 10. Which age group is ranking which genre the most?

# In[101]:

# we are dividing people in 8 group by age.
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
# add new column with age_group
data['Age_group'] = pd.cut(data['Age'], range(0, 81, 10), right=False, labels=labels)
print data[['Age', 'Age_group']].drop_duplicates()[:20]


# In[95]:

# let's look at how age is distributed amongst our users
# call panda's "hist" on the column to produce a histogram
data.Age.hist(bins=10)
plt.title("Distribution of users' ages")
plt.ylabel('count of users')
plt.xlabel('age');


# In[96]:

# now we are comparing ratings across age groups
print data.groupby('Age_group').agg({'Rating': [np.size, np.mean]})


# What we see:
# 
# - old users are more critical than other.

# In[97]:

# group our data by Genres and Age_group
age_genres = data.pivot_table('Rating', rows='Genres', cols='Age_group', aggfunc = 'count')


# In[98]:

# call a function to join genres
group_genres(age_genres)


# In[99]:

# aggregate genres by sum of each age group
age_genres_count = age_genres.groupby("Genres1", as_index=False)
age_genres_count = age_genres_count.aggregate(sum)


# In[100]:

age_genres_count


# In[299]:

# we are using pandas area-plot for better visualization
age_genres_count.plot(kind = 'area', x = age_genres_count.Genres1, figsize=[15, 6])


##### 11. What are the 25 most rated movies?

# In[129]:

# we are grouping movies by title
most_rated = data.groupby('Title').size().order(ascending=False)[:25]
print most_rated


##### 12. Which movies are most highly rated?

# In[103]:

# group data by title and fing rating's mean
movie_stats = data.groupby('Title').agg({'Rating': [np.size, np.mean]})
print movie_stats.sort([('Rating', 'mean')], ascending=False).head()


# In[104]:

# for better analysis we only look at movies that have been rated at least 100 times
atleast_100 = movie_stats['Rating'].size >= 100
print movie_stats[atleast_100].sort([('Rating', 'mean')], ascending=False)[:15]


##### 13. Build Regression.

# In[161]:

# get data and group it by movies
movies = data[['Title', 'MovieID']]
movies_count = movies.groupby("Title", as_index=False).size()
movies_count = pd.DataFrame(movies_count)


# In[165]:

# removing year from the title
def get_year(df):
    for i,item in enumerate(df.index):
        year = item[-5:len(item)-1]
        df['Year'] = year
    for i,item in enumerate(df.index):
        year = item[-5:len(item)-1]
        df['Year'][i] = year  


# In[166]:

# call function
get_year(movies_count)


# In[173]:

# sort by the most popular 
movies_count = movies_count.sort_index(by=0, ascending=False)
movies_count [:10]


# In[197]:

# reindex data to leave only number of movies and years
movies_count.index = range(len(movies_count))
# rename movies
movies_count.columns = ["num_watches", 'year']
movies_count[:10]


# In[234]:

movies_count[["num_watches", 'year']].describe()


# In[238]:

# get values of number and years to build regression
y = np.array(movies_count['num_watches'], dtype=float)
x = np.array(movies_count['year'], dtype=float)


# In[260]:

# we are using numpy and pylab library to get coefficients and build plot 

# fit with np.polyfit
m, b = np.polyfit(x, y, 1)
m, b


# In[243]:

# plot x, y and line
plt.plot(x, y, '.')
plt.plot(x, m*x + b, '-')


# In[262]:

# let's try use other library, for example scipy and build regression 
from scipy import stats


# In[263]:

# get statistic values 
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

print 'r value', r_value
print  'p_value', p_value
print 'standard deviation', std_err

# build plot
line = slope*x+intercept
plot(x,y,'o')
plot(x,line,'r-')




##### 14. Test for significance.

# In[271]:

# descriptive Statistics
print x.max(), x.min()


# In[272]:

print x.mean(), x.var()


# In[273]:

#some sample properties compare to their theoretical counterparts
sstr = 'mean = %6.4f, variance = %6.4f, skew = %6.4f, kurtosis = %6.4f'
m, v, s, k = stats.t.stats(10, moments='mvsk')
print sstr %(m, v, s ,k)


# In[274]:

n, (smin, smax), sm, sv, ss, sk = stats.describe(x)
print sstr %(sm, sv, ss, sk)


# In[ ]:

# in our case the sample statistics differ a by a small amount from their theoretical counterparts.


# In[275]:

# we are using  the t-test to test whether the mean of our sample differs in a statistcally significant way from the theoretical expectation.
print 't-statistic = %6.3f pvalue = %6.4f' %  stats.ttest_1samp(x, m)


# In[281]:

# the Kolmogorov-Smirnov test can be used to test the hypothesis that the sample comes from the standard t-distribution
print 'KS-statistic D = %6.3f pvalue = %6.4f' % stats.kstest(x, 't', (10,))


# In both cases p-value = 0, it means that the data comes from a Normal distribution.

# In[1]:

import seaborn as sns


# In[2]:

import seaborn


# In[ ]:



