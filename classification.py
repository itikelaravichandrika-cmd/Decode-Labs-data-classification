# Project 2 - Data Classification Using AI

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset
iris = load_iris()

X = iris.data
y = iris.target

# 2. Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create the classification model
model = DecisionTreeClassifier(random_state=42)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Data Classification Using AI")
print("--------------------------------")
print("Training data:", len(X_train))
print("Testing data:", len(X_test))
print("Model: Decision Tree Classifier")
print("Accuracy:", accuracy)

# 7. Test with new flower data
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("Predicted flower:", iris.target_names[prediction[0]])