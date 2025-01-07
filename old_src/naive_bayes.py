import json
import math

class NaiveBayesBinaryClassifier:
    def __init__(self, alpha=1.0):
        """
        Initialize the binary Naive Bayes classifier.

        Args:
            alpha (float): Laplace smoothing parameter.
        """
        self.alpha = alpha
        self.class_priors = {}     # {class_label: float}
        self.feature_log_probs = {} # {class_label: [log probabilities for each feature]}
        self.num_features = 0
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier.

        Args:
            X (list of list of int): Training data. Each element is a frequency vector representing a sample.
            y (list of int): Class labels for each sample. Should be 0 or 1 in this binary setting.
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty training data.")
        if len(X) != len(y):
            raise ValueError("Mismatched X and y lengths.")
        
        self.num_features = len(X[0])
        classes = set(y)
        if len(classes) != 2:
            raise ValueError("This classifier is binary, but did not receive exactly 2 distinct class labels.")

        # Separate the data by class
        data_by_class = {c: [] for c in classes}
        for xi, yi in zip(X, y):
            data_by_class[yi].append(xi)
        
        # Compute class priors: P(y=class)
        total_samples = len(y)
        self.class_priors = {c: math.log(len(data_by_class[c]) / total_samples) for c in classes}

        # Compute feature probabilities per class using Laplace smoothing
        # For Multinomial NB:
        # P(feature_i | class) = (sum(feature_i counts in class) + alpha) / (total_counts_in_class + alpha * num_features)
        
        self.feature_log_probs = {}
        for c in classes:
            # Sum up frequencies for each feature in this class
            feature_sums = [0] * self.num_features
            total_count = 0
            for vec in data_by_class[c]:
                for i, val in enumerate(vec):
                    feature_sums[i] += val
                    total_count += val
            
            # Compute log probabilities with Laplace smoothing
            log_probs = []
            for count in feature_sums:
                # Apply Laplace smoothing
                p = (count + self.alpha) / (total_count + self.alpha * self.num_features)
                log_probs.append(math.log(p))
            
            self.feature_log_probs[c] = log_probs
        
        self.is_fitted = True

    def predict_proba(self, x):
        """
        Compute the posterior probabilities for the two classes.

        Args:
            x (list of int): Frequency vector for a single sample.

        Returns:
            dict: {class_label: probability}
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")

        # Compute log posterior for each class
        # log P(y=c|x) ∝ log P(y=c) + Σ x_i * log P(feature_i|c)
        # We return normalized probabilities (exponentiate and normalize)
        
        log_posteriors = {}
        for c in self.class_priors:
            log_p = self.class_priors[c]
            for i, val in enumerate(x):
                log_p += val * self.feature_log_probs[c][i]
            log_posteriors[c] = log_p
        
        # Convert log_posteriors to probabilities
        max_log = max(log_posteriors.values())
        exps = {c: math.exp(lp - max_log) for c, lp in log_posteriors.items()}
        sum_exp = sum(exps.values())
        probs = {c: val / sum_exp for c, val in exps.items()}

        return probs

    def predict(self, x):
        """
        Predict the class for a single sample.

        Args:
            x (list of int): Frequency vector for a single sample.

        Returns:
            int: Predicted class (0 or 1).
        """
        probs = self.predict_proba(x)
        # Return class with highest probability
        return max(probs, key=probs.get)

    def save(self, file_path):
        """
        Save the trained model parameters to a JSON file.

        Args:
            file_path (str): The path to the JSON file where the model will be saved.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model.")
        
        state = {
            "alpha": self.alpha,
            "class_priors": self.class_priors,
            "feature_log_probs": self.feature_log_probs,
            "num_features": self.num_features,
            "is_fitted": self.is_fitted
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=4)

    def load(self, file_path):
        """
        Load the model parameters from a JSON file into the current instance.

        Args:
            file_path (str): The path to the JSON file from which to load the model.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.alpha = state["alpha"]
        self.class_priors = state["class_priors"]
        self.feature_log_probs = state["feature_log_probs"]
        self.num_features = state["num_features"]
        self.is_fitted = state["is_fitted"]


# Example usage:
if __name__ == "__main__":
    # Let's say we have some training data: X as frequency vectors and y as binary labels
    X_train = [
        [2, 0, 1],  # sample 1 features
        [0, 1, 2],  # sample 2 features
        [3, 0, 0],  # sample 3 features
        [0, 2, 1]   # sample 4 features
    ]
    y_train = [0, 1, 0, 1]

    clf = NaiveBayesBinaryClassifier(alpha=1.0)
    clf.fit(X_train, y_train)
    print("Class priors (log):", clf.class_priors)
    print("Feature log probs for each class:", clf.feature_log_probs)

    # Predict on a new sample
    x_test = [1, 1, 1]
    prediction = clf.predict(x_test)
    print("Predicted class for {}: {}".format(x_test, prediction))

    # Save the model
    clf.save("nb_model.json")

    # Create a new classifier and load the saved state
    clf_loaded = NaiveBayesBinaryClassifier()
    clf_loaded.load("nb_model.json")

    # Predict with the loaded classifier
    prediction_loaded = clf_loaded.predict(x_test)
    print("Predicted class with loaded model for {}: {}".format(x_test, prediction_loaded))
