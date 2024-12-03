import argparse
import pickle
from cuisine_classifier import CuisineClassifier
from recipe_db import RecipeDB
import matplotlib.pyplot as plt


def main(retrain_classifier, reload_recipe_db):
    if retrain_classifier:
        print("Retraining classifier...")
        cuisine_classifier = CuisineClassifier("data/yumly_train.json")
        with open("cuisine_classifier.pkl", "wb") as f:
            pickle.dump(cuisine_classifier, f)
    else:
        with open("cuisine_classifier.pkl", "rb") as f:
            cuisine_classifier = pickle.load(f)

    if reload_recipe_db:
        print("Reloading recipe database...")
        recipe_db = RecipeDB(cuisine_classifier)
        with open("recipe_db.pkl", "wb") as f:
            pickle.dump(recipe_db, f)
    else:
        with open("recipe_db.pkl", "rb") as f:
            recipe_db = pickle.load(f)

    principle_components = cuisine_classifier.cuisine_pca
    plt.figure(figsize=(10, 8))
    plt.scatter(principle_components["1"], principle_components["2"])

    for i, txt in enumerate(principle_components.index):
        plt.annotate(
            txt, (principle_components["1"].iloc[i], principle_components["2"].iloc[i])
        )

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("PCA Scatter Plot of Ingredients by Cuisine")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrain_classifier", action="store_true", help="Retrain the classifier"
    )
    parser.add_argument(
        "--reload_recipe_db", action="store_true", help="Retrain the classifier"
    )
    args = parser.parse_args()

    main(
        retrain_classifier=args.retrain_classifier,
        reload_recipe_db=args.reload_recipe_db,
    )
