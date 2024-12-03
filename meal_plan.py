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
        
    def _get_current_ingredients(self):
        """Get set of all ingredients currently in the meal plan."""
        ingredients = set()
        for meal in self.meals:
            ingredients.update(meal['ingredients'].split())
        return ingredients
        
    def _count_new_ingredients(self, recipe_ingredients, current_ingredients):
        """Count how many new ingredients a recipe would add.
        
        Args:
            recipe_ingredients: String of space-separated ingredients
            current_ingredients: Set of existing ingredients
            
        Returns:
            int: Number of new ingredients that would be added
        """
        recipe_ingredient_set = set(recipe_ingredients.split())
        return len(recipe_ingredient_set - current_ingredients)
        
    def _select_random_meal(self):
        """Select a random meal from the recipe database.
        
        Returns:
            dict: Selected recipe with index and details
        """
        idx = random.randint(0, len(self.recipe_db.recipes) - 1)
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
        """Add a meal that introduces the least number of new ingredients."""
        if not self.meals:
            return self.add_first_meal()
            
        current_ingredients = self._get_current_ingredients()
        
        # Calculate new ingredients needed for each recipe
        min_new_ingredients = float('inf')
        best_meal = None
        
        for idx, recipe in self.recipe_db.recipes.iterrows():
            new_ingredient_count = self._count_new_ingredients(
                recipe['ingredients'], 
                current_ingredients
            )
            
            if new_ingredient_count < min_new_ingredients:
                min_new_ingredients = new_ingredient_count
                best_meal = {
                    'index': idx,
                    'title': recipe['title'],
                    'ingredients': recipe['ingredients'],
                    'instructions': recipe['instructions']
                }
        
        if best_meal:
            self.meals.append(best_meal)
            return best_meal
        return None
