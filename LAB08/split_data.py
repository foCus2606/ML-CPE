from sklearn.model_selection import train_test_split

def split_dataset(images, labels, paths, test_size=0.20, val_size=0.10, random_state=42):
    """
    Split data into 70% train, 10% validation, and 20% test.
    Stratification keeps the class proportions similar in each split.
    """
    train_val_x, test_x, train_val_y, test_y, train_val_paths, test_paths = train_test_split(
        images, labels, paths,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    val_ratio_from_train_val = val_size / (1.0 - test_size)
    train_x, val_x, train_y, val_y, train_paths, val_paths = train_test_split(
        train_val_x, train_val_y, train_val_paths,
        test_size=val_ratio_from_train_val,
        random_state=random_state,
        stratify=train_val_y
    )

    return (train_x, train_y, train_paths), (val_x, val_y, val_paths), (test_x, test_y, test_paths)
