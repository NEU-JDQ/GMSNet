

from pathlib import Path


TASK_CONFIG = {
    'task1': {
        'name': 'informative',
        'num_classes': 2,
        'label_map': {
            'informative': 1,
            'not_informative': 0
        }
    },
    'task2': {
        'name': 'humanitarian',
        'num_classes': 8, 
        'label_map': {
            'infrastructure_and_utility_damage': 0,
            'not_humanitarian': 1,
            'other_relevant_information': 2,
            'rescue_volunteering_or_donation_effort': 3,
            'vehicle_damage': 4,
            'affected_individuals': 5,
            'injured_or_dead_people': 6,
            'missing_or_found_people': 7,
        }

    },
    'task3': {
            'name': 'damage',  
            'num_classes': 3,
            'label_map': {
                'severe_damage': 0,
                'mild_damage': 1,
                'little_or_no_damage': 2
            }
    }
}




PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = str(PROJECT_ROOT.parent / 'datasets' / 'settingA')