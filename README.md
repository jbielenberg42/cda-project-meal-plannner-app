# Meal Planner

This is a skeleton code project that mostly just demonstrates what *could* be done with a meal
planning app, which emphasizes the tradeoff between:

* The diversity of the meals
* The effort involved cooking all those meals.

This is created for a term project for the Georgia Tech OMSA program - ISYE 6740 (A great class to
take). The original intent was to create this meal planner, and the algorithm that supports it for
my term project. However, I needed to pivot a bit to mostly focus on the cuisine classifier, in the
name of keeping the project within the realm of feasibility, given my calendar.

This code is a bit "spaghetti code" ish, and I heavily used LLMs (shoutout to https://aider.chat, a
really great tool) just to get something working that exemplified the idea. Even though I ended up
not implementing the full algorithm I had envisioned, I wanted to post the code to support some
notes about how the work I did complete could end up supporting a final version of this app.

## Usage

The first time running, make sure to

1) Create a `data` directory
2) Download
   [this](https://www.kaggle.com/datasets/kaggle/recipe-ingredients-dataset?select=train.json) file,
   and store it as `data/yumly_train.json`
3) Download the recipe box dataset from [Eight Portions](https://eightportions.com/Recipes)
  * [Direct download link](https://eightportions.com/recipes_raw.zip)
4) Unzip the contents into `data`
5) Run the app using `$ python3 app.py --retrain_classifier --reload_recipe_db`

It will take a few minutes running the first time. After that you can remove those flags for much
faster interactions (it saves these objects as pickle files). Unless of course, you want to retrain
the classifier or reload the recipe DB.

Similar to above - there's definitley better ways to support these kinds of details for letting
somone close and run this repo. But I'm mostly just posting this code for visiblity for now.
