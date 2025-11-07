import pandas as pd

file_path = "data/Nutritional_Database_Ciqual.csv"

def load_nutritional_data(path):
    try:
        df = pd.read_csv(
            path,
            sep=",",
            encoding="utf-8",
            quotechar='"',
            header=0,
        )

        # Fix comma decimals (e.g. "0,4" → 0.4) in all numeric columns
        df.replace(",", ".", regex=True, inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

        print("✅ CSV parsed and decimal commas converted!\n")
        print(df[["alim_nom_eng", "Protein (g/100g)", "Carbohydrate (g/100g)", "Fat (g/100g)", "Energy, Regulation EU No 1169/2011 (kcal/100g)"]].head())

    except Exception as e:
        print("❌ Failed to load CSV:", e)

if __name__ == "__main__":
    load_nutritional_data(file_path)