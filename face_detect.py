
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Load the dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
X = lfw_people.data
y = lfw_people.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract features using PCA
n_components = 100
pca = PCA(n_components=n_components, whiten=True)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Build the SVM model
svm = SVC(kernel='rbf', gamma='scale', C=1.0)

# Train the SVM model
svm.fit(X_train_pca, y_train)

# Evaluate the SVM model
score = svm.score(X_test_pca, y_test)
print('Accuracy:', score)
