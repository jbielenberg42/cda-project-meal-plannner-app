import argparse
import pickle
from cuisine_classifier import CuisineClassifier


def main(retrain_classifier):
    if retrain_classifier:
        print("Retraining classifier...")
        cuisine_classifier = CuisineClassifier("data/yumly_train.json")
        with open("cuisine_classifier.pkl", "wb") as f:
            pickle.dump(cuisine_classifier, f)
    else:
        with open("cuisine_classifier.pkl", "rb") as f:
            cuisine_classifier = pickle.load(f)
    
    print(cuisine_classifier.classification_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrain_classifier", action="store_true", help="Retrain the classifier"
    )
    args = parser.parse_args()

    main(retrain_classifier=args.retrain_classifier)
