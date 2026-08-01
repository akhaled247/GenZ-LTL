import argparse
import os

import pandas as pd
from tqdm import tqdm

from evaluation.simulate import simulate
from evaluation.simulate_ma_sar import simulate_ma_sar, TRAIN_ENV as MASAR_TRAIN_ENV
import multiprocessing as mp


env_to_tasks = {
    'PointLtlSafety2-v0': [
        '(F blue) & (!blue U (green & F yellow))',
        '!(magenta | yellow) U (blue & F green)',
        '!green U ((blue | magenta) & (!green U yellow))',
        '((green | blue) => (!yellow U magenta)) U yellow',

        'F (green & (!(blue | yellow) U (magenta & (!yellow U blue)))) & F (!green U yellow)',
        'F ((blue | green) & (!yellow U (blue & (!green U magenta)))) & F (yellow & (!blue U green))',
        '!(magenta | yellow) U (blue & (!green U (yellow & F (green & (!blue U magenta)))))',
        'F (blue & (!yellow U (green & F (yellow & (!(magenta | green) U blue)))))',
    ],

    'PointLtlSafety3-v0': [
        '!(red | cyan) U (blue & (!(yellow | green) U (magenta & (!(cyan | yellow) U green))))',
        '!(green | red) U (cyan & (!(blue | magenta) U (yellow & (!(cyan | red) U blue))))',
        '!(yellow | cyan) U (red & (!(green | blue) U (magenta & (!(red | yellow) U green))))',
    ],

    'PointLtlSafety4-v0': [
        '!(orange | red) U (cyan & (!(blue | green) U (yellow & (!(purple | blue) U magenta))))',
        '!(purple | cyan) U (orange & (!(red | yellow) U (green & (!(magenta | red) U blue))))',
        '!(yellow | purple) U (green & (!(cyan | orange) U (blue & (!(magenta | green) U orange))))',
    ],

    'PointLtlSafety5-v0': [
        '!(teal | magenta) U (orange & (!(lime | blue) U (yellow & (!(green | purple) U red))))',
        '!(cyan | purple) U (magenta & (!(red | teal) U (lime & (!(orange | yellow) U green))))',
        '!(blue | green) U (yellow & (!(teal | lime) U (purple & (!(magenta | red) U orange))))',
    ],

    'LetterSafetyEnv-v0': [
        'F (a & (!b U c)) & F d',
        '(F d) & (!f U (d & F b))',
        '!a U (b & (!c U (d & (!e U f))))',
        '((a | b | c | d) => F (e & (F (f & F g)))) U (h & F i)'

        'F (d & (!(a | b) U (i & (!e U c)))) & F (!(f | g | h) U a)',
        'F ((k & (!b | c U f)) & (!(a | e | h) U g)) & F d',
        '!(j | b | d) U (a & (!c U (f & F (g & (!d U e)))))',
        '!(f | g) U ((a & (!b U c)) & F (d & (!e U f)))',
    ],

    'PointLTL0MASAR1WC-v0': [
        '(!surface_0 U entrapped_0) & F surface_0',
    ],

    'PointLTL0MASAR2WC-v0': [
        '(!surface_0 U entrapped_0) & F surface_0',
        '(!surface_1 U entrapped_1) & F surface_1',
        '((!surface_0 U entrapped_0) & F surface_0) & ((!surface_1 U entrapped_1) & F surface_1)',
        '((!surface_0 & !surface_1) U all_entrapped) & (F surface_0 & F surface_1)',
        '((!surface_0 & !surface_1) U all_entrapped) & F all_surface',
        '(!entrapped_0 U surface_0) & (!entrapped_1 U surface_1) & F (surface_0 & surface_1)',
    ],

    'PointLTL1MASAR2WC-v0': [
            '(!surface_0 U entrapped_0) & F surface_0',
            '(!surface_1 U entrapped_1) & F surface_1',
            '((!surface_0 U entrapped_0) & F surface_0) & ((!surface_1 U entrapped_1) & F surface_1)',
            '((!surface_0 & !surface_1) U all_entrapped) & (F surface_0 & F surface_1)',
            '((!surface_0 & !surface_1) U all_entrapped) & F all_surface',
            '(!entrapped_0 U surface_0) & (!entrapped_1 U surface_1) & F (surface_0 & surface_1)',
        ],

    'PointLTL2MASAR2WC-v0': [
            '(!surface_0 U entrapped_0) & F surface_0',
            '(!surface_1 U entrapped_1) & F surface_1',
            '((!surface_0 U entrapped_0) & F surface_0) & ((!surface_1 U entrapped_1) & F surface_1)',
            '((!surface_0 & !surface_1) U all_entrapped) & (F surface_0 & F surface_1)',
            '((!surface_0 & !surface_1) U all_entrapped) & F all_surface',
            '(!entrapped_0 U surface_0) & (!entrapped_1 U surface_1) & F (surface_0 & surface_1)',
        ],
}


def main(env, exp, seeds):
    num_episodes = 100
    tasks = env_to_tasks[env]
    gamma = {
        'PointLtl2-v0': 0.998,
        'PointLtlSafety2-v0': 0.998,
        'PointLtlSafety3-v0': 0.998,
        'PointLtlSafety4-v0': 0.998,
        'PointLtlSafety5-v0': 0.998,
        'LetterEnv-v0': 0.94,
        'LetterSafetyEnv-v0': 0.94,
        'PointLTL0MASAR1WC-v0': 0.998,
        'PointLTL0MASAR2WC-v0': 0.998,
        'PointLTL1MASAR1WC-v0': 0.998,
        'PointLTL1MASAR2WC-v0': 0.998,
        'PointLTL2MASAR1WC-v0': 0.998,
        'PointLTL2MASAR2WC-v0': 0.998,
    }[env]
    
    use_ma_eval = env == 'PointLTL0MASAR2WC-v0'
    num_procs = len(seeds)
    with mp.Pool(num_procs) as pool:
        args = []
        for task in tasks:
            print(f'Running task: {task}')
            for seed in seeds:
                if use_ma_eval:
                    args.append([env, MASAR_TRAIN_ENV, gamma, exp, seed, num_episodes, task, False, True])
                else:
                    args.append([env, gamma, exp, seed, num_episodes, task, True, False, True])
        eval_fn = eval_ma_task if use_ma_eval else eval_task
        for result in tqdm(pool.imap_unordered(eval_fn, args), total=len(seeds)*len(tasks)):
            results = []
            (sn, vn, mean_steps), seed, task = result
            results.append([exp, task, seed, sn, vn, mean_steps])
            df = pd.DataFrame(results, columns=['method', 'task', 'seed', 'success_rate', 'violation_rate', 'mean_steps'])
            os.makedirs('results_finite', exist_ok=True)

            file_path = f'results_finite/{env}_{exp}.csv'
            if os.path.exists(file_path):
                df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                df.to_csv(file_path, index=False)


def eval_task(simulate_args):
    return simulate(*simulate_args), simulate_args[3], simulate_args[5]


def eval_ma_task(simulate_args):
    eval_env, train_env, gamma, exp, seed, num_episodes, formula, render, deterministic = simulate_args
    return (
        simulate_ma_sar(
            eval_env, train_env, gamma, exp, seed, num_episodes, formula, render, deterministic,
        ),
        seed,
        formula,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=['LetterSafetyEnv-v0',
                                                    'PointLtlSafety2-v0',
                                                    'PointLtlSafety3-v0',
                                                    'PointLtlSafety4-v0',
                                                    'PointLtlSafety5-v0',
                                                    'PointLTL0MASAR1WC-v0',
                                                    'PointLTL0MASAR2WC-v0'], default='PointLtlSafety2-v0')
    parser.add_argument('--exp', type=str, default='GenZ-LTL')
    parser.add_argument('--seed', type=int, nargs='+', default=[1], help="List of seeds")
    args = parser.parse_args()
    main(args.env, args.exp, args.seed)
