import json
from pathlib import Path


def _get_resource_path():
    return Path(__file__).parent.parent.parent


def generate_daily_budget():
    return 60.0


def generate_strategies():
    resource_path = _get_resource_path()
    strategies_path = resource_path.joinpath('resource/strategies.json')
    with open(strategies_path, 'rb') as f:
        return json.load(f)


def generate_business_constrains():
    resource_path = _get_resource_path()
    business_constraints_path = resource_path.joinpath('resource/business_constraints.json')
    with open(business_constraints_path, 'rb') as f:
        return json.load(f)
