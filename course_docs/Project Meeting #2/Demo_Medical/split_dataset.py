import numpy as np


# TODO: MORE IN-DEPTH Exploring Data Analysis (EDA) - and make sure that all clients have okish distrbutions to learn from 

def save_split_per_clients(dataset, num_clients=2):
    data_train, labels_train = dataset["train_images"], dataset["train_labels"]
    
    # Concatenate val images and labels to training set
    if "val_images" in dataset and "val_labels" in dataset:
        data_val, labels_val = dataset["val_images"], dataset["val_labels"]
        data_train = np.concatenate((data_train, data_val), axis=0)
        labels_train = np.concatenate((labels_train, labels_val), axis=0)

    client_datasets_train = np.array_split(data_train, num_clients)
    client_labels_train = np.array_split(labels_train, num_clients)

    data_test, labels_test = dataset["test_images"], dataset["test_labels"]
    client_datasets_test = np.array_split(data_test, num_clients)
    client_labels_test = np.array_split(labels_test, num_clients)
    
    for i in range(num_clients):
        np.savez_compressed(f"client_{i+1}_data.npz", train_images=client_datasets_train[i], train_labels=client_labels_train[i],
                                                         test_images=client_datasets_test[i], test_labels=client_labels_test[i],
                                                         train_mean=np.mean(client_datasets_train[i]), train_std=np.std(client_datasets_train[i]))


#


def show_datasets_info_per_client(num_clients=2):
    for i in range(num_clients):
        data = np.load(f"client_{i+1}_data.npz")
        tr_images, tr_labels = data["train_images"], data["train_labels"]
        te_images, te_labels = data["test_images"], data["test_labels"]
        print(f"Client {i+1}:")
        print(f"  Train images: {tr_images.shape}, Train labels: {tr_labels.shape}")
        print(f"  Test images: {te_images.shape}, Test labels: {te_labels.shape}")
        print(f"  Train mean: {data['train_mean']:.4f}, Train std: {data['train_std']:.4f}")


if __name__ == "__main__":
    dataset = np.load("pneumoniamnist.npz")

    print(f"Dataset entries: {list(dataset.keys())}")
    print(dataset["train_images"].shape)  # (3882, 28, 28)
    print(dataset["test_images"].shape)   # ( 624, 28, 28)

    save_split_per_clients(dataset, num_clients=3)
    show_datasets_info_per_client(num_clients=3)