import subprocess
import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/humaid_combined.csv")

def ask_gemma(prompt):
    result = subprocess.run(
        ["ollama", "run", "gemma3n"],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()

def build_zero_shot_prompt(tweet):
    return f"""
Classify this tweet into ONE of the following EXACT categories:

caution_and_advice
displaced_people_and_evacuations
infrastructure_and_utility_damage
injured_or_dead_people
missing_or_found_people
not_humanitarian
other_relevant_information
requests_or_urgent_needs
rescue_volunteering_or_donation_effort
sympathy_and_support

Return ONLY the exact label.

Tweet: "{tweet}"
"""

predictions = []

# change number here if needed
N = 100

for tweet in df["tweet_text"][:N]:
    prompt = build_zero_shot_prompt(tweet)
    output = ask_gemma(prompt)
    predictions.append(output)
    print("Tweet:", tweet)
    print("Prediction:", output)
    print("-" * 50)

# Create subset
df_subset = df.head(N).copy()
df_subset["prediction"] = predictions

# Save results
df_subset.to_csv("results/gemma_zero_shot.csv", index=False)

print("Saved to results/gemma_zero_shot.csv")

# Accuracy
print("\nAccuracy check:")

correct = 0
for i, row in df_subset.iterrows():
    if row["prediction"] == row["class_label"]:
        correct += 1

accuracy = correct / len(df_subset)
print(f"Accuracy: {accuracy:.2f}")