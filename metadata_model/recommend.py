import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import faiss


text_model = SentenceTransformer('all-MiniLM-L6-v2')
scaler = MinMaxScaler()
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')


class MetadataMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=256):
        super(MetadataMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch Norm
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.output = nn.Linear(128, output_dim)
        self.relu = nn.LeakyReLU()  # Better than ReLU
        self.dropout = nn.Dropout(0.3)  # Prevent overfitting

    def forward(self, x):
        x = self.fc1(x)

        if x.shape[0] > 1:  # Apply BatchNorm only if batch size > 1
            x = self.bn1(x)

        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        if x.shape[0] > 1:
            x = self.bn2(x)

        x = self.relu(x)
        x = self.output(x)
        return x


class MetadataDataset(Dataset):
    def __init__(self, df, categorical_features, text_features):
        self.df = df
        self.categorical_features = categorical_features
        self.text_features = text_features
        self.text_data = preprocess_text(df, text_features)
        self.categorical_data = preprocess_categorical(
            df, categorical_features)
        self.labels = self.create_pairs()

    def __len__(self):
        return len(self.df)

    def create_pairs(self):
        """
        Creates labels for contrastive learning. Similar pairs have label 1, dissimilar pairs -1.
        """
        labels = []
        for i in range(len(self.df) // 2):
            labels.append(1)  # Similar pair
            labels.append(-1)  # Dissimilar pair
        return torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, idx):
        input_tensor = torch.cat(
            (self.categorical_data[idx], self.text_data[idx]), dim=0)
        paired_idx = (idx + 1) % len(self.df)  # Pair with the next artwork
        paired_tensor = torch.cat(
            (self.categorical_data[paired_idx], self.text_data[paired_idx]), dim=0)
        return input_tensor, paired_tensor, self.labels[idx]


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
    df = df.copy()
    missing_cols = [
        col for col in categorical_columns if col not in df.columns]
    for col in missing_cols:
        df[col] = "Unknown"

    cat_cols = df[categorical_columns]  # Ensure consistent order
    encoded = one_hot_encoder.fit_transform(
        cat_cols) if fit else one_hot_encoder.transform(cat_cols)

    return torch.tensor(encoded, dtype=torch.float32)


def preprocess_text(df, text_columns, batch_size=64):
    """
    Encode text columns of a dataframe using a sentence transformer model.

    Args:
        df (pd.DataFrame): dataframe containing text columns
        text_columns (list): names of text columns to encode

    Returns:
        torch.tensor: tensor of encoded text columns
    """
    df['combined_text'] = df[text_columns].fillna("").agg(" ".join, axis=1)

    embeddings_list = []
    for i in range(0, len(df), batch_size):
        batch = df['combined_text'].iloc[i: i + batch_size].tolist()
        batch_embeddings = text_model.encode(batch, convert_to_tensor=True)
        embeddings_list.append(batch_embeddings)

    return torch.cat(embeddings_list, dim=0)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def train(model, dataloader, epochs=100, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = ContrastiveLoss(margin=0.5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for input_tensor, paired_tensor, labels in dataloader:
            input_tensor, paired_tensor, labels = input_tensor.to(
                device), paired_tensor.to(device), labels.to(device)
            optimizer.zero_grad()
            embedding_a = model(input_tensor)
            embedding_b = model(paired_tensor)
            loss = criterion(embedding_a, embedding_b, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "metadata_mlp_trained.pth")
    print("Training complete. Model saved.")


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, embedding_a, embedding_b, label):
        return self.loss_fn(embedding_a, embedding_b, label)


def visualize_embeddings(embeddings, labels=None, method='TSNE', n_components=2):
    """
    Visualizes the embeddings in 2D using dimensionality reduction (t-SNE or PCA).

    Args:
        embeddings (numpy.ndarray): The embeddings to visualize (shape: [n_samples, n_features]).
        labels (numpy.ndarray, optional): Labels to color the points in the plot.
        method (str, optional): The dimensionality reduction method ('TSNE' or 'PCA'). Defaults to 'TSNE'.
        n_components (int, optional): The number of components for reduction. Defaults to 2.
    """
    if method == 'TSNE':
        reducer = TSNE(n_components=n_components)
    elif method == 'PCA':
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError("Method should be 'TSNE' or 'PCA'.")

    # Reduce the dimensions of the embeddings
    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))

    if labels is not None:
        scatter = plt.scatter(
            reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
    else:
        plt.scatter(reduced_embeddings[:, 0],
                    reduced_embeddings[:, 1], alpha=0.6)

    plt.title('Embeddings Visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig('test.png')


def create_index():
    """
    Create a Faiss index for efficient nearest neighbor search.

    Returns:
        faiss.IndexFlatL2: Index for storing and searching 128-dimensional vectors.
    """
    index = faiss.IndexFlatL2(256)  # L2 Normalization
    return index


def query_similar_artworks(index, query_embedding, k=5):
    query_embedding = query_embedding / \
        np.linalg.norm(query_embedding)  # Normalize
    query_embedding = query_embedding.cpu().detach().numpy().reshape(1, -1)
    _, indices = index.search(query_embedding, k)
    return indices


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
    query_embedding = query_embedding.cpu().detach().numpy().reshape(1, -1)
    _, indices = index.search(query_embedding, k)
    return indices


def find_similar_artworks(new_artwork, model, index, df, categorical_features, text_features):
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
    text_data = preprocess_text(new_df, text_features)

    # Combine features and generate embedding
    input_tensor = torch.cat((categorical_data, text_data), dim=1)
    input_tensor = input_tensor / input_tensor.norm(dim=1, keepdim=True)

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient tracking
        query_embedding = model(input_tensor)

    # Find similar artworks
    similar_indices = query_similar_artworks(index, query_embedding, k=5)
    similar_artworks = df.iloc[similar_indices[0]]
    return similar_artworks[text_features + categorical_features]


if __name__ == '__main__':
    df = pd.read_csv('all_met_data_with_dates.csv', sep=',', low_memory=False)

    text_features = ['title', 'artist']
    categorical_features = ['medium', 'department',
                            'culture', 'period', 'classification']
    df = df.drop_duplicates(subset=text_features + categorical_features)
    categorical_data = preprocess_categorical(
        df, categorical_features, fit=True)
    text_data = preprocess_text(df, text_features)

    # Dataset & Dataloader
    dataset = MetadataDataset(df, categorical_features, text_features)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Combine all features
    final_dataset = torch.cat((categorical_data, text_data), dim=1)
    input_dim = final_dataset.shape[1]

    # Initialize Model
    input_dim = categorical_data.shape[1] + text_data.shape[1]
    model = MetadataMLP(input_dim=input_dim)
    model.apply(initialize_weights)

    # Train Model
    train(model, dataloader, epochs=50, lr=0.001)
    embedding = model(final_dataset)
    metadata_embeddings_np = embedding.cpu().detach().numpy()
    np.save("metadata_embeddings.npy", metadata_embeddings_np)
    torch.save(model.state_dict(), "metadata_mlp.pth")

    # FAISS index setup
    index = create_index()
    index.add(metadata_embeddings_np)

    # Example New Artwork for Prediction
    new_artwork = {
        'title': 'Train Landscape',
        'artist': 'Ellsworth Kelly',
        'medium': 'Oil on canvas; three joined panels',
        'department': 'Contemporary Art',
        'culture': 'United States',
        'period': '1953',
        'classification': 'painting'
    }

    # Find and display the top 5 similar artworks
    similar_artworks = find_similar_artworks(
        new_artwork, model, index, df, categorical_features, text_features)
    print(similar_artworks)

    department_labels = df['department'].astype('category').cat.codes.values
    visualize_embeddings(metadata_embeddings_np,
                         labels=department_labels, method='TSNE', n_components=2)
