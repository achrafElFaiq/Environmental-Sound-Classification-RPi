import matplotlib.pyplot as plt

def extract_data(filename):
    """
    Extract epoch numbers and accuracy values from the given file.

    Parameters:
    filename (str): Path to the file containing training logs.

    Returns:
    tuple: Two lists containing epoch numbers and accuracy values.
    """
    epochs = []
    accuracies = []
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            epoch_counter = 1
            for line in lines:
                if 'accuracy' in line:
                    epochs.append(epoch_counter)
                    try:
                        accuracy = float(line.split()[-1])
                        accuracies.append(accuracy)
                    except ValueError:
                        print(f"Skipping line due to invalid accuracy value: {line.strip()}")
                    epoch_counter += 1
    except FileNotFoundError:
        print(f"File {filename} not found.")
    return epochs, accuracies

def plot_accuracy(epochs, accuracies):
    """
    Plot accuracy values against epoch numbers.

    Parameters:
    epochs (list): List of epoch numbers.
    accuracies (list): List of accuracy values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, marker='o', linestyle='-', color='b')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    filename = 'learning.txt'
    epochs, accuracies = extract_data(filename)
    if epochs and accuracies:
        plot_accuracy(epochs, accuracies)
    else:
        print("No data to plot.")

