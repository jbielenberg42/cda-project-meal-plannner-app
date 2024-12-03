from recipe_db import RecipeDB
from cuisine_classifier import CuisineClassifier
import random


class MealPlan:
    def __init__(self, recipe_db: RecipeDB, cuisine_classifier: CuisineClassifier):
        """Initialize a meal plan with recipe database and cuisine classifier.

        Args:
            recipe_db: RecipeDB object containing all available recipes
            cuisine_classifier: CuisineClassifier object for cuisine predictions
        """
        self.recipe_db = recipe_db
        self.cuisine_classifier = cuisine_classifier
        self.meals = []
        self.used_indices = set()
        self.selected_cuisines = set()

    def _get_current_ingredients(self):
        """Get set of all ingredients currently in the meal plan."""
        meal_idxs = [meal["index"] for meal in self.meals]
        selected_meal_ingredients = self.recipe_db.recipe_ingredient_matrix.iloc[meal_idxs]
        return set(selected_meal_ingredients.columns[selected_meal_ingredients.sum() > 0])

    def _select_random_meal(self):
        """Select a random meal from the recipe database.

        Returns:
            dict: Selected recipe with index and details
        """
        available_indices = set(range(len(self.recipe_db.recipes))) - self.used_indices
        if not available_indices:
            raise ValueError("No more available meals to select from!")

        idx = random.choice(list(available_indices))
        self.used_indices.add(idx)
        recipe = self.recipe_db.recipes.iloc[idx]
        return {
            "index": idx,
            "title": recipe["title"],
            "ingredients": self.recipe_db.get_recipe_ingredient_features(idx),
            "instructions": recipe["instructions"],
        }

    def add_first_meal(self):
        """Add the first randomly selected meal to the meal plan."""
        if not self.meals:  # Only add if meals list is empty
            meal = self._select_random_meal()
            self.meals.append(meal)
            return meal
        return None

    def _get_target_cuisine(self):
        """Determine the cuisine with maximum total distance from selected cuisines."""
        if not self.selected_cuisines:
            return random.choice(list(self.cuisine_classifier.cuisine_distances.index))
        cuisine_distances = self.cuisine_classifier.avg_dist_from_other_cuisines(
            self.selected_cuisines
        )
        return cuisine_distances.idxmax()

    def add_optimal_meal(self):
        """Add a meal that introduces the least number of new ingredients from the target cuisine.

        Returns:
            tuple: (meal_dict, set of new ingredients added, predicted meal cuisine)
        """
        if not self.meals:
            meal = self.add_first_meal()
            cuisine = self.recipe_db.recipe_cuisine_predictions.iloc[meal["index"]][
                "predicted_cuisine"
            ]
            self.selected_cuisines.add(cuisine)
            return (
                meal,
                self.recipe_db.get_recipe_ingredient_features(meal["index"]),
                cuisine,
            )

        target_cuisine = self._get_target_cuisine()
        current_ingredients = self._get_current_ingredients()

        adjusted_ingredient_matrix = self.recipe_db.recipe_ingredient_matrix.drop(
            columns = list(current_ingredients)
        )
        new_ingredients_per_recipe = (adjusted_ingredient_matrix).sum(axis=1)

        # Filter for recipes from target cuisine
        cuisine_mask = (
            self.recipe_db.recipe_cuisine_predictions["predicted_cuisine"]
            == target_cuisine
        )
        available_mask = (
            ~new_ingredients_per_recipe.index.isin(self.used_indices) & cuisine_mask
        )

        if not available_mask.any():
            # If no recipes available from target cuisine, fall back to any cuisine
            available_mask = ~new_ingredients_per_recipe.index.isin(self.used_indices)

        best_idx = new_ingredients_per_recipe[available_mask].idxmin()
        self.used_indices.add(best_idx)
        best_recipe = self.recipe_db.recipes.iloc[best_idx]

        # Update selected cuisines
        cuisine = self.recipe_db.recipe_cuisine_predictions.iloc[best_idx][
            "predicted_cuisine"
        ]
        self.selected_cuisines.add(cuisine)

        best_meal = {
            "index": best_idx,
            "title": best_recipe["title"],
            "ingredients": self.recipe_db.get_recipe_ingredient_features(best_idx),
            "instructions": best_recipe["instructions"],
        }

        # Calculate new ingredients this meal adds using the ingredient matrix
        new_ingredients = set(
            adjusted_ingredient_matrix.columns[
                adjusted_ingredient_matrix.iloc[best_idx] > 0
            ]
        )

        self.meals.append(best_meal)
        return best_meal, new_ingredients, cuisine

    def plan_meals(self, num_meals=4):
        """Plan multiple meals while minimizing new ingredients.

        Args:
            num_meals: Number of meals to plan (default: 4)

        Returns:
            list: List of tuples containing (meal info, new ingredients)
        """
        meals = []
        all_ingredients = set()

        for i in range(num_meals):
            meal, new_ingredients, cuisine = self.add_optimal_meal()
            meals.append((meal, new_ingredients, cuisine))
            all_ingredients.update(new_ingredients)

        return meals, all_ingredients
