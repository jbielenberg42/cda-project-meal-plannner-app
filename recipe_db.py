import pandas as pd
import json


class RecipeDB:

    def __init__(self):
        self.recipes = self.load_and_clean_data()

    def load_and_clean_data(self):
        with open("data/recipes_raw_nosource_epi.json", "r") as json_f:
            epicurious_df = pd.DataFrame(json.load(json_f)).T.reset_index(drop=True)

        with open("data/recipes_raw_nosource_fn.json", "r") as json_f:
            fn_df = pd.DataFrame(json.load(json_f)).T.reset_index(drop=True)

        with open("data/recipes_raw_nosource_ar.json", "r") as json_f:
            ar_df = pd.DataFrame(json.load(json_f)).T.reset_index(drop=True)
        df = pd.concat([epicurious_df, fn_df, ar_df], ignore_index=True)
        df = df[["title", "instructions", "ingredients"]]
        df = df.dropna().reset_index(drop=True)
        df["ingredients"] = df["ingredients"].apply(
            lambda x: " ".join(x) if isinstance(x, list) else x
        )
        return df
