import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_reviews = 200
dates = pd.to_datetime(['2023-01-15'] * num_reviews) #Example, can be expanded
ratings = np.random.randint(1, 6, size=num_reviews)
aspects = ['Food Quality', 'Service', 'Ambiance', 'Value', 'Cleanliness']
review_aspects = [np.random.choice(aspects) for _ in range(num_reviews)]
review_text = [f"The {np.random.choice(aspects)} was {np.random.choice(['excellent','good','average','poor','terrible'])}" for _ in range(num_reviews)]
df = pd.DataFrame({
    'Date': dates,
    'Rating': ratings,
    'Aspect': review_aspects,
    'Review': review_text
})
#Simulate rating volatility by adding some random noise.
df['Rating'] = df['Rating'] + np.random.normal(0, 0.5, size=num_reviews)
df['Rating'] = df['Rating'].clip(1,5) #Ensure ratings stay within 1-5 range.
# --- 2. Data Cleaning and Preparation ---
#No significant cleaning needed for synthetic data, but this section is crucial for real-world datasets.
# --- 3. Analysis ---
# Group data by aspect and calculate average rating for each aspect.
average_ratings_by_aspect = df.groupby('Aspect')['Rating'].mean()
#Correlation analysis between aspects and rating volatility (simplified example)
#In a real scenario, more sophisticated time series analysis would be needed.
correlation_matrix = df.groupby('Aspect')['Rating'].agg(['mean', 'std']).corr()
# --- 4. Visualization ---
plt.figure(figsize=(10, 6))
sns.barplot(x=average_ratings_by_aspect.index, y=average_ratings_by_aspect.values)
plt.title('Average Rating by Aspect')
plt.xlabel('Aspect of Restaurant')
plt.ylabel('Average Rating')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('average_rating_by_aspect.png')
print("Plot saved to average_rating_by_aspect.png")
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation between Aspects and Rating Statistics')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
print("Plot saved to correlation_matrix.png")