### Modified from ChatGPT output
# Let's write a Python script to separate the annotations file into train, val, and test annotation files.

# File paths for input files
annotations_file_path = '/expanse/lustre/projects/ddp390/akulkarni/datasets/list_attr_celeba.txt'  # Replace with your actual file path
split_file_path = '/expanse/lustre/projects/ddp390/akulkarni/datasets/list_eval_partition.txt'  # Replace with your actual file path

# Output file paths
train_output_path = '/expanse/lustre/projects/ddp390/akulkarni/datasets/celeba64_train_annotations.txt'
val_output_path = '/expanse/lustre/projects/ddp390/akulkarni/datasets/celeba64_val_annotations.txt'
test_output_path = '/expanse/lustre/projects/ddp390/akulkarni/datasets/celeba64_test_annotations.txt'

# Dictionaries to hold annotations
annotations_dict = {}

# Reading the annotations file
with open(annotations_file_path, 'r') as annotations_file:
    for line in annotations_file:
        line = line.strip()
        if line:
            parts = line.split()
            filename = parts[0]
            labels = parts[1:]
            annotations_dict[filename] = labels

# Open output files for writing
with open(train_output_path, 'w') as train_file, open(val_output_path, 'w') as val_file, open(test_output_path, 'w') as test_file:
    # Reading the split file and writing to the respective files
    with open(split_file_path, 'r') as split_file:
        for line in split_file:
            line = line.strip()
            if line:
                parts = line.split()
                filename = parts[0]
                split_type = parts[1]

                # Get the annotation line for the current filename
                if filename in annotations_dict:
                    annotation_line = f"{filename} {' '.join(annotations_dict[filename])}\n"

                    # Write to the corresponding output file
                    if split_type == '0':  # Train
                        train_file.write(annotation_line)
                    elif split_type == '1':  # Val
                        val_file.write(annotation_line)
                    elif split_type == '2':  # Test
                        test_file.write(annotation_line)

print("Annotation files have been separated into train, val, and test files successfully.")
