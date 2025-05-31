import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from phinet_model import train_phinet_model 

# Load dataset
df = pd.read_csv('email_data.csv', on_bad_lines='skip',engine='python')
df.columns = df.columns.str.strip().str.lower()

# Rename to match PHINet input
df.rename(columns={
    'url': 'urls',
    'attached_file': 'attachments',
    'label': 'label'
}, inplace=True)

# Fill missing URL/attachment values
df['urls'] = df['urls'].fillna('')
df['attachments'] = df['attachments'].fillna('')

# Clean and prepare data
df = df.dropna(subset=['email_body', 'label'])
df = df[df['label'].isin(['Phishing Email', 'Legitimate Email'])]
df['label'] = df['label'].map({'Phishing Email': 1, 'Legitimate Email': 0})

# Check if dataset is large enough and contains both classes
if df['label'].nunique() < 2 or df.shape[0] < 5:
    print("❌ Not enough labeled data to train. Add more data to email_data.csv.")
    print("Current label distribution:\n", df['label'].value_counts())
    exit()

print("✅ Data label distribution:\n", df['label'].value_counts())

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# ✅ Train model and get feature_engine
model, feature_engine = train_phinet_model(train_df)

# Save to current directory
base_path = os.path.dirname(os.path.abspath(__file__))

# Save model and feature engine
with open(os.path.join(base_path, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)
with open(os.path.join(base_path, 'feature_engine.pkl'), 'wb') as f:
    pickle.dump(feature_engine, f)

# Evaluate
X_test = feature_engine.transform(test_df)
y_test = test_df['label']
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# 🔍 Optional: Debugging insight
print("\n🔍 Sample phishing features:")
print(feature_engine.transform(df[df['label'] == 1].head(1)))
print("\n🔍 Sample legitimate features:")
print(feature_engine.transform(df[df['label'] == 0].head(1)))
