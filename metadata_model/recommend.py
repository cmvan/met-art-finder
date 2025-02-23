import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sentence_transformers import SentenceTransformer
import faiss

text_model = SentenceTransformer('all-MiniLM-L6-v2')
scaler = MinMaxScaler()
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


class MetadataMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super(MetadataMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.output = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.output(x)


def preprocess_numerical(df, numerical_columns, fit=False):
    """
    Scales numerical columns of a dataframe using MinMaxScaler.

    Args:
        df (pd.DataFrame): dataframe containing numerical columns
        numerical_columns (list): names of numerical columns to scale
        fit (bool, optional): Whether to fit scaler to data or not. Defaults to False.

    Returns:
        torch.tensor: tensor of scaled numerical columns
    """
    if fit:
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    else:
        df[numerical_columns] = scaler.transform(df[numerical_columns])
    return torch.tensor(df[numerical_columns].values, dtype=torch.float32)


def preprocess_categorical(df, categorical_columns, fit=False):
    """
    One-hot encodes categorical columns of a dataframe using OneHotEncoder.

    Args:
        df (pd.DataFrame): dataframe containing categorical columns
        categorical_columns (list): names of categorical columns to one-hot encode
        fit (bool, optional): Whether to fit one-hot encoder to data or not. Defaults to False.

    Returns:
        torch.tensor: tensor of one-hot encoded categorical columns
    """
    if fit:
        encoded = one_hot_encoder.fit_transform(df[categorical_columns])
    else:
        encoded = one_hot_encoder.transform(df[categorical_columns])
    return torch.tensor(encoded, dtype=torch.float32)


def preprocess_text(df, text_columns):
    """
    Encode text columns of a dataframe using a sentence transformer model.

    Args:
        df (pd.DataFrame): dataframe containing text columns
        text_columns (list): names of text columns to encode

    Returns:
        torch.tensor: tensor of encoded text columns
    """
    df['combined_text'] = df[text_columns].fillna("").agg(" ".join, axis=1)
    embeddings = text_model.encode(
        df['combined_text'].tolist(), convert_to_tensor=True)
    return embeddings  # Already a tensor


def create_index():
    """
    Create a Faiss index for efficient nearest neighbor search.

    Returns:
        faiss.IndexFlatL2: Index for storing and searching 128-dimensional vectors.
    """
    return faiss.IndexFlatL2(128)


def query_similar_artworks(index, query_embedding, k=5):
    """
    Query a Faiss index to find the k most similar artworks to a given embedding.

    Args:
        index (faiss.Index): Index containing 128-dimensional vectors of artworks.
        query_embedding (torch.tensor): 128-dimensional vector of the input artwork.
        k (int, optional): Number of similar artworks to return. Defaults to 5.

    Returns:
        numpy.ndarray: Indices of the k most similar artworks in the index.
    """
    distances, indices = index.search(
        query_embedding.cpu().detach().numpy(), k)
    return indices


def find_similar_artworks(new_artwork, model, index, df, categorical_features, numerical_features, text_features):
    """
    Find the k most similar artworks to a given input artwork.

    Args:
        new_artwork (dict): Dictionary containing the features of the input artwork.
        model (MetadataMLP): Model used to generate embeddings.
        index (faiss.Index): Index containing 128-dimensional vectors of artworks.
        df (pd.DataFrame): Dataframe containing the artwork metadata.
        categorical_features (list): Names of categorical columns to one-hot encode.
        numerical_features (list): Names of numerical columns to scale.
        text_features (list): Names of text columns to encode using a sentence transformer model.

    Returns:
        pd.DataFrame: Dataframe containing the k most similar artworks to the input artwork.
    """
    new_df = pd.DataFrame([new_artwork])

    # Preprocess input
    categorical_data = preprocess_categorical(new_df, categorical_features)
    numerical_data = preprocess_numerical(new_df, numerical_features)
    text_data = preprocess_text(new_df, text_features)

    # Combine features and generate embedding
    input_tensor = torch.cat(
        (categorical_data, numerical_data, text_data), dim=1)
    query_embedding = model(input_tensor)

    # Find similar artworks
    similar_indices = query_similar_artworks(index, query_embedding, k=5)
    similar_artworks = df.iloc[similar_indices[0]]
    return similar_artworks[['Title', 'Artist Display Name', 'Medium', 'Culture',
                             'Original Object Begin Date', 'Original Object End Date']]


if __name__ == '__main__':
    df = pd.read_csv('cleaned_met_dataset.csv', sep=',', low_memory=False)

    numerical_features = ['Object Begin Date', 'Object End Date']
    text_features = ['Object Name', 'Title',
                     'Artist Display Name', 'Artist Display Bio', 'Dimensions']
    categorical_features = ['Department', 'Culture',
                            'Medium', 'Classification', 'Artist Role']

    # Store original dates before normalization
    df['Original Object Begin Date'] = df['Object Begin Date']
    df['Original Object End Date'] = df['Object End Date']

    # Fit preprocessors
    categorical_data = preprocess_categorical(
        df, categorical_features, fit=True)
    numerical_data = preprocess_numerical(df, numerical_features, fit=True)
    text_data = preprocess_text(df, text_features)

    # Combine all features
    final_dataset = torch.cat(
        (categorical_data, numerical_data, text_data), dim=1)
    input_dim = final_dataset.shape[1]

    # Define and run MLP model
    model = MetadataMLP(input_dim=input_dim)
    embedding = model(final_dataset)

    # FAISS index setup
    index = create_index()
    index.add(embedding.cpu().detach().numpy())

    # Example New Artwork for Prediction
    new_artwork = {
        "Department": "European Paintings",
        "Culture": "French",
        "Medium": "Oil on canvas",
        "Classification": "Paintings",
        "Artist Role": "Artist",
        "Object Begin Date": 1884,
        "Object End Date": 1884,
        "Object Name": "Painting",
        "Title": "View of Bordighera",
        "Artist Display Name": "Claude Monet",
        "Artist Display Bio": "French, 1884 Bordighera",
        "Dimensions": "60 cm x 73 cm"
    }

    # Find and display the top 5 similar artworks
    similar_artworks = find_similar_artworks(
        new_artwork, model, index, df, categorical_features, numerical_features, text_features)
    print(similar_artworks)
