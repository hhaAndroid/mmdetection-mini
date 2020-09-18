from mmdet.cv_core.utils.env import collect_env as collect_base_env


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')