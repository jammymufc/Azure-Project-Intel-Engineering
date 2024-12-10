import pandas as pd

# Load your dataset
df = pd.read_csv("C:\\Users\\USER\\Documents\\MSc AI\\Intelligence Engineering\\Project\\Watch_accelerometer.csv")

# List of all activity types
activities = ['bike', 'sit', 'stairsdown', 'stairsup', 'stand', 'walk']

# Number of samples to extract per activity
samples_per_activity = 1200

# Initialize an empty list to store the extracted data
extracted_data = []

# Loop through each user
for user in df['User'].unique():
    user_data = df[df['User'] == user]
    
    # Loop through each activity type
    for activity in activities:
        activity_data = user_data[user_data['gt'] == activity]
        
        # Ensure there are enough rows for sampling
        if len(activity_data) >= samples_per_activity:
            # Randomly sample `samples_per_activity` rows
            sampled_activity_data = activity_data.sample(n=samples_per_activity, random_state=42)
            
            # Extract only the necessary columns (x, y, z, Arrival_Time, etc.)
            selected_columns = sampled_activity_data[['x', 'y', 'z', 'gt']]
            extracted_data.append(selected_columns)

# Convert the list of DataFrames into a single DataFrame
final_extracted_data = pd.concat(extracted_data, axis=0)

# Optionally, save the final extracted data to a CSV file
final_extracted_data.to_csv('extracted_data.csv', index=False)
