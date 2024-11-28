from cuisine_classifier import CuisineClassifier

def main():
    # Initialize classifier with training data
    classifier = CuisineClassifier("data/yumly_train.json")
    
    # Print out the out-of-bag score (accuracy)
    print(f"Classifier accuracy (OOB score): {classifier.cuisine_clf.oob_score_:.3f}")

if __name__ == "__main__":
    main()
