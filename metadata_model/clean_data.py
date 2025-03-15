import pandas as pd
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
            return "Unknown"
        nationalities = set(n.strip() for n in nat_str.split(
            "|") if n.strip())  # Deduplicate & trim
        return ", ".join(nationalities) if nationalities else "Unknown"

    # Apply cleaning
    df[column_name] = df[column_name].apply(process_nationalities)

    # Flatten and count nationality occurrences
    all_nationalities = [
        nat for sublist in df[column_name].str.split(", ") for nat in sublist]
    nationality_counts = Counter(all_nationalities)

    # Select the top N most common nationalities (others will be grouped under "Other")
    top_nationalities = set(
        [n for n, _ in nationality_counts.most_common(top_n)])

    # Function to replace rare nationalities with "Other"
    def filter_nationalities(nat_str):
        nationalities = nat_str.split(", ")
        filtered = [
            nat if nat in top_nationalities else "Other" for nat in nationalities]
        return ", ".join(filtered)

    df[column_name] = df[column_name].apply(filter_nationalities)

    return df, top_nationalities


def main():
    df = pd.read_csv('met_dataset.csv', sep=',', low_memory=False)

    numerical_features = ['Object Begin Date', 'Object End Date']
    text_features = ['Object Name', 'Title',
                     'Artist Display Name', 'Artist Display Bio', 'Dimensions']
    categorical_features = ['Department', 'Culture', 'Medium',
                            'Classification', 'Artist Nationality', 'Period', 'Country']

    cols = numerical_features + text_features + categorical_features
    df = df[cols]

    # Drop rows missing both Title & Medium
    df.dropna(subset=['Title', 'Medium'], how='all', inplace=True)

    # Remove rows with too many missing values
    df = df[df.isnull().sum(axis=1) < 4]

    print("Dataset shape after cleaning:", df.shape)

    # Fill missing values
    df.loc[:, 'Title'] = df['Title'].fillna(df['Object Name'])
    df.loc[:, 'Culture'] = df.groupby(['Artist Display Name', 'Medium', 'Period'])['Culture'] \
        .transform(lambda x: x.agg(lambda y: y.mode().iat[0] if not y.mode().empty else "Unknown Culture"))
    df.loc[:, 'Artist Nationality'] = df['Artist Nationality'].fillna(
        df['Country'])
    df.loc[:, 'Artist Display Bio'] = df['Artist Display Bio'].fillna(
        "Biography Unavailable")
    # df.loc[:, 'Artist Role'] = df['Artist Role'].fillna("Role Unknown")
    df.loc[:, 'Dimensions'] = df['Dimensions'].fillna(
        "Dimensions Not Available")
    df.loc[:, 'Medium'] = df['Medium'].fillna("Unknown Medium")
    df.loc[:, 'Classification'] = df['Classification'].fillna("Unclassified")

    df['Dimensions'] = df['Dimensions'].str.replace(
        r'[\n\r\t]+', ' ', regex=True).str.strip()
    df['Dimensions'] = df['Dimensions'].str.replace(
        r'\s+', ' ', regex=True).str.strip()

    # Convert text and categorical columns to string
    for col in text_features + categorical_features:
        df[col] = df[col].astype(str)

    # Convert numerical columns to numeric
    for column in numerical_features:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Clean artist roles
    # df, _ = clean_artist_roles(df)

    # Clean artist nationalities
    df, _ = clean_artist_nationality(df)

    df.drop(columns=['Country', 'Object Name', 'Period'], inplace=True)

    # Save cleaned dataset to csv
    df.to_csv('cleaned_met_2_dataset.csv', index=False)


if __name__ == '__main__':
    main()
