import pandas as pd


def analyze_datasets(train_path, dev_path):
    print(f"Loading dataset splits...\n")
    try:
        train_df = pd.read_csv(train_path, sep='\t')
        dev_df = pd.read_csv(dev_path, sep='\t')
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return

    print("=== 1. Dataset sizes ===")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(dev_df)}\n")

    label_col = next((col for col in train_df.columns if 'label' in col.lower() or 'class' in col.lower()), None)
    img_col = next((col for col in train_df.columns if 'image' in col.lower() or 'img' in col.lower()), None)
    text_col = next((col for col in train_df.columns if 'text' in col.lower() or 'tweet' in col.lower()), None)

    if label_col:
        print(f"=== 2. Validation label distribution (column: '{label_col}') ===")
        print(dev_df[label_col].value_counts())
        print("\nValidation class proportions:")
        print(dev_df[label_col].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
        print()

    if img_col:
        print(f"=== 3. Train-validation image overlap (column: '{img_col}') ===")
        train_imgs = set(train_df[img_col].dropna())
        dev_imgs = set(dev_df[img_col].dropna())
        overlap = train_imgs.intersection(dev_imgs)
        print(f"Validation image identifiers also present in training: {len(overlap)}")
        if len(dev_imgs) > 0:
            print(f"Overlap proportion: {len(overlap) / len(dev_imgs):.2%}\n")

    if text_col:
        print(f"=== 4. Train-validation text overlap (column: '{text_col}') ===")
        train_texts = set(train_df[text_col].dropna())
        dev_texts = set(dev_df[text_col].dropna())
        overlap_texts = train_texts.intersection(dev_texts)
        print(f"Validation text values also present in training: {len(overlap_texts)}")
        if len(dev_texts) > 0:
            print(f"Overlap proportion: {len(overlap_texts) / len(dev_texts):.2%}\n")


if __name__ == "__main__":
    analyze_datasets('task_damage_text_img_train.tsv', 'task_damage_text_img_dev.tsv')