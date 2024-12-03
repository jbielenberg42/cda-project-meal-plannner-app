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
