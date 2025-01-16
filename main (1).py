""" Emily Chae
    ITP-449
    HW 12
    Description: separate the different wine qualities of red wine
"""

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def main():
    # write your code here

    # 1. Import the dataset into a DataFrame.

    df = pd.read_csv("wineQualityReds.csv")

    # 2. Store quality in a separate Series and drop it from the 
    # original DataFrame. You will use this later to compare the 
    # results to.

    quality = df['quality']

    # 3. Drop quality from the DataFrame.

    df = df.drop('quality', axis=1)

    # 4. Normalize all columns of the DataFrame.

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_normalized = pd.DataFrame(df_scaled, columns=df.columns)

    # 5. Instantiate, train, and extract inertia_ values from 10 
    # K-Means Clustering models, one each from k of 1 to k of 10. 
    # Use: random_state = 42

    inertia_ = []

    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_normalized)
        inertia_.append(kmeans.inertia_)

    # 6. Plot the chart of inertia vs number of clusters k 
    # (see sample output).

    plt.plot(range(1, 11), inertia_, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Inertia vs Number of Clusters')
    plt.show()

    # 7. What is the "optimal" k (number of clusters) based 
    # on the generated plot?

    print("7. Since this plot appears to elbow at k = 4, this is the best number of clusters")
    print()
    # 8. Cluster the wines a final time into the "optimal" 
    # number of k clusters. Use: random_state = 42

    optimal_k = 4
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans_optimal.fit_predict(df_normalized)

    # 9. Extract the cluster numbers (also known as labels_) 
    # from the model and combine it with the quality Series 
    # to form a new DatFrame. Call this DataFrame results.

    results = pd.DataFrame({'Cluster': clusters, 'Quality': quality})

    # 10. Show the crosstab of cluster number vs quality.

    crosstab_result = pd.crosstab(results['Cluster'], results['Quality'])
    print('10.')
    print(crosstab_result)
    print()

    # 11. Given the data in the crosstab: do the clusters represent 
    # the quality of wine? Why or why not?


    print("11.")
    print("Cluster 0 has a high concentration of quality category 5 wines and a decent number in category 6, but very few high-quality wines (7 and 8).")
    print("Cluster 1 seems to represent a more balanced distribution across medium to high-quality wines, with significant numbers in categories 5, 6, and 7, and some in 8, suggesting a range of medium to high quality.")
    print("Cluster 2 is heavily weighted towards lower quality wines, with a majority in categories 5 and a significant number in category 4, but almost none in the highest quality category (8).")
    print("Cluster 3 has its majority in category 6, followed by 7 and a reasonable count in 8, suggesting that it represents higher quality wines well.")
    print("Thus, Cluster 0 and 2 try to capture lower quality wines. Cluster 1 tries to capture mid-range wines. Cluster 3 tries to capture higher quality wines. These clusters are representative in capturing patterns (albeit the patterns are a bit vague); that said, while there are patterns, the clusters aren't perfectly homogenous with respect to the specific wine quality.") 


    plt.savefig('Red wine clustering.png')




if __name__ == '__main__':
    main()
