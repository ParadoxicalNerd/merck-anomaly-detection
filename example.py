import pandas as pd

# For syntax highlighting in VSCode, add src to pylance path

from src.XML_to_SQL import XML_to_SQL
from src.Healthkit import Healthkit
from src.SQL_Interface import SQL

db_path = "database/db.sqlite"
dataset_path = "../datasets/Apple Healthkit data/"

# Creates a connection to an sql database
conn, cur = SQL(db_path).create_connection()

# Reads exported zips from Apple Health kit and add it to the database
healthkit_reader = XML_to_SQL(dataset_path)
healthkit_reader.zips_to_db(conn, cur)

# Read the SQL data into a pandas data frame
healthcare_records = pd.read_sql_query("SELECT * FROM healthkit_records", conn)

# Automatically clean the pandas dataset
hk = Healthkit()
cleaned_data = hk.clean_data(healthcare_records)

# Plot the cleaned histograms
# hk.plot_cleaned_histograms(cleaned_data)

# Merges all user data to one dataset and cacluates metrics based on pre-generated models
# If you do not provide a path for the model, it will calcualte the model from the dataset
df = hk.merge_users(cleaned_data)
X_PCA = hk.calcuate_pca(df)
# X_PCA = hk.calcuate_pca(df, "models/pca_model.pkl")
dist = hk.calculate_dist_metric(X_PCA)
# dist = hk.calculate_dist_metric(X_PCA, "models/dist_metric_model.pkl")

# Exports model to file
hk.export_pca('models/pca_metric.pkl')
hk.export_dist_metric('models/dist_metric.pkl')

# Makes 3d plot of the complexity reduced data and overlays it with the distance
# Note: Needs browser to work
hk.plot_3d_points(X_PCA, dist, "scatter_plot.html")

# Calculates and prints the anomaly threshold
# Optional: Pass it how many standard deviations away you want the threshold to be at
threshold = hk.caclulate_threshold(dist)
print(threshold)
