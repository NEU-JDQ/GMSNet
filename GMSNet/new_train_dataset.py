import os
import pandas as pd


def process_datasets(base_path, add_ratio, random_seed=42):
    """Create modified splits by transferring sampled test rows into training.

    Args:
        base_path: Directory containing the original TSV splits.
        add_ratio: Fraction of test rows to transfer into training.
        random_seed: Random seed used for sampling.

    The resulting files use the _new suffix and change the held-out split."""

    tasks = ['damage', 'humanitarian', 'informative']

    print(f"Creating modified splits; fraction transferred from test to training: {add_ratio * 100}%")

    for task in tasks:
        train_file = os.path.join(base_path, f'task_{task}_text_img_train.tsv')
        test_file = os.path.join(base_path, f'task_{task}_text_img_test.tsv')

        new_train_file = os.path.join(base_path, f'task_{task}_text_img_train_new.tsv')
        new_test_file = os.path.join(base_path, f'task_{task}_text_img_test_new.tsv')

        try:
            df_train = pd.read_csv(train_file, sep='\t')
            df_test = pd.read_csv(test_file, sep='\t')
        except FileNotFoundError as e:
            print(f"File not found: {e.filename}. Verify the dataset path.")
            continue

        # Moving test samples into training changes the held-out evaluation protocol.
        df_test_sampled = df_test.sample(frac=add_ratio, random_state=random_seed)

        df_test_new = df_test.drop(df_test_sampled.index)

        df_train_new = pd.concat([df_train, df_test_sampled], ignore_index=True)


        df_train_new.to_csv(new_train_file, sep='\t', index=False)
        df_test_new.to_csv(new_test_file, sep='\t', index=False)

        print(f"Task [{task}]:")
        print(f"  - Original training samples: {len(df_train)}")
        print(f"  - Original test samples: {len(df_test)}")
        print(f"  - Transferred samples:   {len(df_test_sampled)}")
        print(f"  - Modified training split: {new_train_file} (samples: {len(df_train_new)})")
        print(f"  - Modified test split: {new_test_file} (samples: {len(df_test_new)})\n")


if __name__ == '__main__':
    dataset_path = '/home/tSdu/_New_World/xzh/crisiskan/datasets/settingA'

    RATIO_TO_MOVE = 0.45

    process_datasets(dataset_path, RATIO_TO_MOVE)