import argparse
from colorama import Fore

from utils.logging.logger import Logger


class TextLogger(Logger):
    """
    A logger that logs to standard output.
    """

    def __init__(self, config: argparse.Namespace):
        super().__init__(config)

    def log_config(self):
        print(self.config.experiment)

    def log(self, data: dict[str, float | list[float]]):
        data = self.aggregate(data)
        self.check_keys_valid(data)
        row = ''
        for key, value in data.items():
            if key == 'avg_goal_success':
                continue
            short_name = self.get_short_name(key)
            row += f'{short_name}: '
            if isinstance(value, float):
                row += f'{value:.2f} | '
            else:
                row += f'{value} | '
        row = row[:-3]  # remove trailing ' | '
        print(row)

    @staticmethod
    def get_short_name(key: str) -> str:
        if key == 'return_per_episode_mean':
            return 'r_mu'
        elif key == 'return_per_episode_std':
            return 'r_std'
        elif key == 'num_steps_per_episode_mean':
            return 's_mu'
        elif key == 'num_steps_per_episode_std':
            return 's_std'
        elif key == 'success_per_episode_mean':
            return 'P_mu'
        elif key == 'success_per_episode_std':
            return 'P_std'
        elif key == 'violation_per_episode_mean':
            return 'V_mu'
        elif key == 'violation_per_episode_std':
            return 'V_std'
        elif key == 'duration':
            return 't'
        else:
            return key

    @staticmethod
    def info(message: str):
        print(message)

    @staticmethod
    def important_info(message: str):
        print(f'{Fore.GREEN}{message}{Fore.RESET}')
