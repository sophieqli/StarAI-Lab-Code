from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Step 1: Fetch the dataset
car_data = fetch_ucirepo(id=19)
X = car_data.data.features
y = car_data.data.targets

# Step 2: Combine features and target
df = pd.concat([X, y], axis=1)

# Step 3: Encode categorical columns as integers
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # Save the encoder for future use if needed

# Step 4: Write to a file with no header and no index
df.to_csv("car_categorical_encoded.txt", index=False, header=False)

