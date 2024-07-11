import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

# File containing the predictions
input_file = 'test.txt'

# Initialize dictionaries to count correct and incorrect predictions per class
correct_counts = defaultdict(int)
incorrect_counts = defaultdict(int)

# Dictionary to track confusions
confusions = defaultdict(lambda: defaultdict(int))

# Initialize counters for total files and correct answers
total_files = 0
correct_answers = 0

# Read the file and process each line
with open(input_file, 'r') as file:
    for line in file:
        if not line.strip():
            continue  # Skip empty lines
        
        parts = line.split(':')
        if len(parts) < 2:
            continue  # Skip lines that don't have the expected format
        
        file_path = parts[0].strip()
        try:
            predicted_class = int(parts[2].strip().split('[')[1].split(']')[0])
        except (IndexError, ValueError):
            continue  # Skip lines with incorrect format for predicted class
        
        # Extract classID from the file name
        try:
            class_id = int(file_path.split('-')[1])
        except (IndexError, ValueError):
            continue  # Skip lines with incorrect format for file path
        
        # Increment the total files counter
        total_files += 1
        
        # Compare and update counts
        if class_id == predicted_class:
            correct_counts[class_id] += 1
            correct_answers += 1
        else:
            incorrect_counts[class_id] += 1
        
        # Update confusion matrix
        confusions[class_id][predicted_class] += 1

# Prepare data for plotting
classes = sorted(set(list(correct_counts.keys()) + list(incorrect_counts.keys())))
correct = [correct_counts[cls] for cls in classes]
incorrect = [incorrect_counts[cls] for cls in classes]

# Plot the bar graph
bar_width = 0.35
index = range(len(classes))

fig, ax = plt.subplots()
bar1 = plt.bar(index, correct, bar_width, label='Correct')
bar2 = plt.bar([i + bar_width for i in index], incorrect, bar_width, label='Incorrect')

plt.xlabel('Class ID')
plt.ylabel('Number of Predictions')
plt.title('Comparison of Actual Class vs Predicted Class')
plt.xticks([i + bar_width / 2 for i in index], classes)
plt.legend()

plt.tight_layout()
plt.show()

# Prepare confusion matrix
confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)

for class_id in classes:
    for predicted_class in classes:
        confusion_matrix[class_id, predicted_class] = confusions[class_id][predicted_class]

# Plot confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')
plt.show()

# Print the number of total files and correct answers
print(f"Total files: {total_files}")
print(f"Correct answers: {correct_answers}")
