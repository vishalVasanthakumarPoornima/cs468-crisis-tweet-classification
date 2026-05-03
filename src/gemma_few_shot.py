import subprocess
import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/humaid_combined.csv")

# Check columns (run once, then you can remove)
print(df.columns)

def ask_gemma(prompt):
    result = subprocess.run(
        ["ollama", "run", "gemma3n"],
        input=prompt,
        text=True,
        capture_output=True
    )
    return result.stdout.strip()

def build_few_shot_prompt(tweet):
    return f"""
Classify the tweet into one of these categories:
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

Examples:
Tweet: "Praying for all families affected"
Answer: sympathy_and_support

Tweet: "Homes destroyed by wildfire"
Answer: infrastructure_and_utility_damage

Tweet: "Donate to help victims"
Answer: rescue_volunteering_or_donation_effort

Now classify:
Tweet: "{tweet}"

Answer:
"""

predictions = []

# Run on small sample first
for tweet in df["tweet_text"][:100]:
    prompt = build_few_shot_prompt(tweet)
    output = ask_gemma(prompt)
    predictions.append(output)
    print("Tweet:", tweet)
    print("Prediction:", output)
    print("-" * 50)

# Save results
df_subset = df.head(100).copy()
df_subset["prediction"] = predictions

df_subset.to_csv("results/gemma_predictions.csv", index=False)

print("Saved to results/gemma_predictions.csv")

df_subset = df.head(100).copy()
df_subset["prediction"] = predictions

df_subset.to_csv("results/gemma_predictions.csv", index=False)

print("Saved to results/gemma_predictions.csv")

# ADD THIS RIGHT HERE
print("\nAccuracy check:")

correct = 0
for i, row in df_subset.iterrows():
    if row["prediction"] == row["class_label"]:
        correct += 1

accuracy = correct / len(df_subset)
print(f"Accuracy: {accuracy:.2f}")