import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Read and parse CSV file
data = []
with open('/home/codemaster29/Documents/Coding_Stuff/Operations_on_Python/weforshe/styles.csv', 'r') as file:
    for line in file:
        # Strip any leading/trailing whitespace and split by comma
        data.append(line.strip().split(','))

# Check if data has at least one row (header) and subsequent rows
if len(data) > 1:
    df = pd.DataFrame(data[1:], columns=data[0])  # Assuming first row is header

    # Step 2: Feature Selection
    features = ['year', 'usage', 'productDisplayName']
    df_selected = df[features]

    # Convert 'year' to numeric (handle errors by coercing to NaN)
    df_selected['year'] = pd.to_numeric(df_selected['year'], errors='coerce')

    # Encode 'usage' column as dummy variables
    df_selected = pd.get_dummies(df_selected, columns=['usage'])

    # Step 3: Clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_selected.drop('productDisplayName', axis=1))

    kmeans = KMeans(n_clusters=12, random_state=42)  # You can adjust the number of clusters
    df_selected['Cluster'] = kmeans.fit_predict(scaled_features)

    # Step 4: Recommendation Function
    def get_recommendations(year, place):
        # Find the cluster for the given input
        input_data = pd.DataFrame([[year, place]], columns=['year', f'usage_{place}'])
        input_scaled = scaler.transform(input_data)
        cluster = kmeans.predict(input_scaled)[0]

        # Get recommendations from the same cluster
        recommendations = df_selected[df_selected['Cluster'] == cluster]['productDisplayName'].tolist()

        return recommendations[:5]  # Return top 5 recommendations

    # Example usage
    year = 2012
    place = 'Casual'
    recommendations = get_recommendations(year, place)
    print(f"Recommendations for year {year} and place '{place}':")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    # Save the processed data to a new CSV file
    df_selected.to_csv('processed_data.csv', index=False)

else:
    print("Error: No data found in 'paste.txt' or incorrect format.")

