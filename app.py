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
    
    # Create meal plan
    meal_plan = MealPlan(recipe_db, cuisine_classifier)
    all_ingredients = set()
    
    print("\nSelecting 4 meals with minimal new ingredients:\n")
    
    # Select 4 meals
    for i in range(4):
        meal, new_ingredients = meal_plan.add_optimal_meal()
        print(f"\nMeal {i+1}: {meal['title']}")
        print(f"New ingredients added ({len(new_ingredients)}): {', '.join(sorted(new_ingredients))}")
        all_ingredients.update(new_ingredients)
    
    print(f"\nTotal unique ingredients needed: {len(all_ingredients)}")


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
