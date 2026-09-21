"""Audit train/dev/test without modifying datasets; no ML packages required."""
import argparse
import json
from data.config import DEFAULT_DATA_ROOT, TASK_CONFIG
from data.protocol import audit_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--task_name', choices=['all', *TASK_CONFIG], default='all')
    parser.add_argument('--hash-images', action='store_true',
                        help='Detect identical image bytes under different filenames')
    args = parser.parse_args()
    tasks = TASK_CONFIG if args.task_name == 'all' else [args.task_name]
    reports = {task: audit_splits(args.data_root, TASK_CONFIG[task], args.hash_images)
               for task in tasks}
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
