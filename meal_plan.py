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
        self.meals = []  # Will store selected meals from recipe_db
        self.used_indices = set()  # Track which meals have been used
        
    def _get_current_ingredients(self):
        """Get set of all ingredients currently in the meal plan."""
        ingredients = set()
        for meal in self.meals:
            ingredients.update(meal['ingredients'].split())
        return ingredients
        
    def _count_new_ingredients_vectorized(self, ingredients_series, current_ingredients):
        """Count new ingredients for each recipe in vectorized way.
        
        Args:
            ingredients_series: Series of space-separated ingredients strings
            current_ingredients: Set of existing ingredients
            
        Returns:
            Series: Number of new ingredients that would be added for each recipe
        """
        return ingredients_series.apply(
            lambda x: len(set(x.split()) - current_ingredients)
        )
        
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
            'index': idx,
            'title': recipe['title'],
            'ingredients': recipe['ingredients'],
            'instructions': recipe['instructions']
        }
        
    def add_first_meal(self):
        """Add the first randomly selected meal to the meal plan."""
        if not self.meals:  # Only add if meals list is empty
            meal = self._select_random_meal()
            self.meals.append(meal)
            return meal
        return None
        
    def add_optimal_meal(self):
        """Add a meal that introduces the least number of new ingredients.
        
        Returns:
            tuple: (meal_dict, set of new ingredients added)
        """
        if not self.meals:
            meal = self.add_first_meal()
            return meal, set(meal['ingredients'].split())
            
        current_ingredients = self._get_current_ingredients()
        
        # Calculate new ingredients needed for all recipes at once
        new_ingredient_counts = self._count_new_ingredients_vectorized(
            self.recipe_db.recipes['ingredients'],
            current_ingredients
        )
        
        # Get index of recipe with minimum new ingredients, excluding used meals
        available_mask = ~new_ingredient_counts.index.isin(self.used_indices)
        best_idx = new_ingredient_counts[available_mask].idxmin()
        self.used_indices.add(best_idx)
        best_recipe = self.recipe_db.recipes.iloc[best_idx]
        
        best_meal = {
            'index': best_idx,
            'title': best_recipe['title'],
            'ingredients': best_recipe['ingredients'],
            'instructions': best_recipe['instructions']
        }
        
        # Calculate new ingredients this meal adds
        new_ingredients = set(best_meal['ingredients'].split()) - current_ingredients
        
        self.meals.append(best_meal)
        return best_meal, new_ingredients

    def plan_meals(self, num_meals=4):
        """Plan multiple meals while minimizing new ingredients.
        
        Args:
            num_meals: Number of meals to plan (default: 4)
            
        Returns:
            list: List of tuples containing (meal info, new ingredients)
        """
        results = []
        all_ingredients = set()
        
        for i in range(num_meals):
            meal, new_ingredients = self.add_optimal_meal()
            results.append((meal, new_ingredients))
            all_ingredients.update(new_ingredients)
            
        return results, all_ingredients
