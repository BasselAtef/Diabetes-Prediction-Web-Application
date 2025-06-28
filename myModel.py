
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle


# Load dataset
df = pd.read_csv('Diabetes.csv', sep='\t')
df.describe()


df['GLU'] = df['GLU'].replace(0, df['GLU'].mean())
df['BP'] = df['BP'].replace(0, df['BP'].mean())
df['ST'] = df['ST'].replace(0, df['ST'].mean())
df['INS'] = df['INS'].replace(0, df['INS'].mean())
df['BMI'] = df['BMI'].replace(0, df['BMI'].mean())

# Split dataset into features (X) and target variable (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Scale data using StandardScaler
scaler= StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Remove outliers using interquartile range (IQR) method
Q1= X.quantile(0.25)
Q3 = X.quantile(0.75)
IQR = Q3 - Q1
outliers = (X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))
X = X[~outliers.any(axis=1)]
y = y[~outliers.any(axis=1)]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create Random Forest classifier model
clf = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)

# Train Random Forest classifier model on training data
clf.fit(X_train, y_train)

# Make predictions on testing data
y_pred = clf.predict(X_test)

# Evaluate model performance
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc}")
print(f"Confusion matrix:\n{cm}")



# dump information to that file
pickle.dump(clf,open('model.pkl','wb'))

clf = pickle.load(open('model.pkl', 'rb'))
