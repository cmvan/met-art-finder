import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def clean_artist_roles(df, column_name="Artist Role", top_n_roles=10):
    # Step 1: Trim whitespace and split on '|'
    df[column_name] = df[column_name].str.strip().str.split("|")

    # Step 2: Flatten list and count occurrences
    all_roles = [role for sublist in df[column_name].dropna()
                 for role in sublist]
    role_counts = pd.Series(all_roles).value_counts()

    # Step 3: Identify top N roles to keep
    top_roles = set(role_counts.head(top_n_roles).index)

    # Step 4: Deduplicate and filter rare roles
    def process_roles(roles):
        if roles is None or not isinstance(roles, list):
            return None
        roles = set(roles)  # Remove duplicates
        # Keep only common roles
        roles = [role for role in roles if role in top_roles]
        # Sort for consistency
        return "|".join(sorted(roles)) if roles else "Other"

    df[column_name] = df[column_name].apply(process_roles)

    return df, role_counts


def clean_artist_nationality(df, column_name="Artist Nationality", top_n=20):
    # Fill NaN values with "Unknown"
    df[column_name] = df[column_name].fillna("Unknown").str.strip()

    # Function to process each entry
    def process_nationalities(nat_str):
        if not isinstance(nat_str, str) or not nat_str.strip():
            return ["Unknown"]
        nationalities = list(set(n.strip() for n in nat_str.split(
            "|") if n.strip()))  # Deduplicate & trim
        return nationalities if nationalities else ["Unknown"]

    # Apply cleaning
    df[column_name] = df[column_name].apply(process_nationalities)

    # Flatten and count nationality occurrences
    all_nationalities = [nat for sublist in df[column_name] for nat in sublist]
    nationality_counts = Counter(all_nationalities)

    # Select the top N most common nationalities (others will be grouped under "Other")
    top_nationalities = set(
        [n for n, _ in nationality_counts.most_common(top_n)])

    # Function to replace rare nationalities with "Other"
    def filter_nationalities(nationalities):
        return [nat if nat in top_nationalities else "Other" for nat in nationalities]

    df[column_name] = df[column_name].apply(filter_nationalities)

    return df, top_nationalities


def main():
    df = pd.read_csv('met_dataset.csv', sep=',', low_memory=False)

    numerical_features = ['Object Begin Date', 'Object End Date']
    text_features = ['Object Name', 'Title',
                     'Artist Display Name', 'Artist Display Bio', 'Dimensions']
    categorical_features = ['Department', 'Culture', 'Medium',
                            'Classification', 'Artist Role', 'Artist Nationality', 'Country']

    df = df[numerical_features + text_features + categorical_features]

    # Clean artist roles
    df, role_stats = clean_artist_roles(df)

    # Clean artist nationalities
    df, nationality_stats = clean_artist_nationality(df)

    # Convert numerical columns to numeric
    for column in numerical_features:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Convert categorical columns to string
    for column in categorical_features:
        df[column] = df[column].astype(str)

    # Fill NaN values with "Unknown"
    df.fillna('Unknown', inplace=True)
    df.replace('nan', 'Unknown', inplace=True)

    # Save cleaned dataset to csv
    df.to_csv('cleaned_met_dataset.csv', index=False)


if __name__ == '__main__':
    main()
