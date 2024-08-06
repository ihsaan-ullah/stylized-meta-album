# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# !/usr/bin/env python

import argparse
import json
import numpy as np
import os
# import submitit
import sys
import time
import torch
import models
import itertools
from datasets import get_loaders
from omegaconf import OmegaConf
from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import TqdmCallback
from deephyper.search.hps import CBO
import ray

from utils import Tee, flatten_dictionary_for_wandb, results_to_log_dict

import wandb


def randl(l_):
    return l_[torch.randperm(len(l_))[0]]

os.environ["WANDB__SERVICE_WAIT"] = "500"


def parse_args():
    parser = argparse.ArgumentParser(description='Training configurations.')
    parser.add_argument('--config', type=str, default='config_deephyper.yaml')
    return parser.parse_args()


def run(job=None):
    """
    Main function to run the training of the model
    :param job: If the job is not None, it means that the function is called by DeepHyper
    else, it is called after with the best hyperparameters found by DeepHyper
    :return: The mean accuracy on the validation set
    """
    project_name = args.wandb_project

    # Start by replacing all the hyperparameters optimized by DeepHyper
    if job is not None:
        for parameters, value in job.parameters.items():
            setattr(args, parameters, value)
    else:
        project_name = args.wandb_project + "_best"

    if 'SMA' not in args:
        args.SMA = None
    wandb.init(project=project_name, config=flatten_dictionary_for_wandb(dict(args)))
    print(args)
    print(f'Method: {args.method}')
    # print(args)
    start_time = time.time()
    torch.manual_seed(args["init_seed"])
    np.random.seed(args["init_seed"])
    loaders = get_loaders(args["data_path"], args["dataset"], int(args["batch_size"]), args["method"], args.SMA)

    sys.stdout = Tee(os.path.join(
        args["output_dir"], f'seed_{args["hparams_seed"]}_{args["init_seed"]}.out'), sys.stdout)
    sys.stderr = Tee(os.path.join(
        args["output_dir"], f'seed_{args["hparams_seed"]}_{args["init_seed"]}.err'), sys.stderr)
    checkpoint_file = os.path.join(
        args["output_dir"], f'seed_{args["hparams_seed"]}_{args["init_seed"]}.pt')
    best_checkpoint_file = os.path.join(
        args["output_dir"],
        f"seed_{args['hparams_seed']}_{args['init_seed']}.best.pt")
    result_file = os.path.join(
        args["output_dir"], f"{args['wandb_project']}.csv")
    model = {
        "erm": models.ERM,
        "suby": models.ERM,
        "subg": models.ERM,
        "rwy": models.ERM,
        "rwg": models.ERM,
        "dro": models.GroupDRO,
        "jtt": models.JTT
    }[args["method"]](args, loaders["tr"])

    last_epoch = 0
    best_selec_val = float('-inf')
    best_mean_group_acc_va = float('-inf')
    best_mean_group_acc_te = float('-inf')
    best_acc_va = float('-inf')
    best_acc_te = float('-inf')
    best_worst_acc_va = float('-inf')
    best_worst_acc_te = float('-inf')
    best_worst3_acc_va = float('-inf')
    best_worst3_acc_te = float('-inf')
    best_worst5_grp_acc_va = float('-inf')
    best_worst5_grp_acc_te = float('-inf')
    # Deactivate loading model for now
    # if os.path.exists(checkpoint_file):
    #     model.load(checkpoint_file)
    #     last_epoch = model.last_epoch
    #     best_selec_val = model.best_selec_val

    for epoch in range(last_epoch, args["num_epochs"]):
        if epoch == args["T"] + 1 and args["method"] == "jtt":
            loaders = get_loaders(
                args["data_path"],
                args["dataset"],
                args["batch_size"],
                args["method"],
                args["SMA"],
                model.weights.tolist())

        for i, x, y, g in loaders["tr"]:
            model.update(i, x, y, g, epoch)

        result = {"epoch": epoch, "time": time.time() - start_time, "lr": model.optimizer.param_groups[0]["lr"]}
        if epoch % args.eval_every_n_epochs == 0:
            for loader_name, loader in loaders.items():
                avg_acc, group_accs = model.accuracy(loader) # 
                result["acc_" + loader_name] = group_accs
                result["avg_acc_" + loader_name] = avg_acc # micro average 
                result["mean_grp_acc_" + loader_name] = np.mean(group_accs) # this one does not take into account the size of the group, this is a "macro" average
                result["worst_acc_" + loader_name] = min(group_accs)
                if args.SMA.K**2 >= 3:
                    result["worst3_acc_" + loader_name] = np.mean(sorted(group_accs)[:3])
                if args.SMA.K**2 >= 5:
                    result["worst5_grp_acc_" + loader_name] = np.mean(sorted(group_accs)[:5])
                
            # print("DEBUG: ", result["acc_va"])
            # print("DEBUG AVG: ", result["avg_acc_va"])
            selec_metrics = {
                "worst_acc_va": min(result["acc_va"]),
                "avg_acc_va": result["avg_acc_va"],
                "mean_grp_acc_va": result["mean_grp_acc_va"],
                
            }
            if args.SMA.K**2 >= 3:
                selec_metrics["worst3_acc_va"] = np.mean(sorted(result["acc_va"])[:3])
            if args.SMA.K**2 >= 5:
                selec_metrics["worst5_grp_acc_va"] = np.mean(sorted(result["acc_va"])[:5])
                
            selec_value = selec_metrics[args["selector"]]

            if selec_value >= best_selec_val:
                model.best_selec_val = selec_value
                best_selec_val = selec_value
                model.save(best_checkpoint_file)

            if result["mean_grp_acc_va"] >= best_mean_group_acc_va:
                best_mean_group_acc_va = result['mean_grp_acc_va']
                best_mean_group_acc_te = result['mean_grp_acc_te']
            if result["avg_acc_va"] >= best_acc_va:
                best_acc_va = result['avg_acc_va']
                best_acc_te = result['avg_acc_te']
            # same with best worst_acc :
            if result["worst_acc_va"] >= best_worst_acc_va:
                best_worst_acc_va = result['worst_acc_va']
                best_worst_acc_te = result['worst_acc_te']
            if args.SMA.K**2 >= 3:
                if result["worst3_acc_va"] >= best_worst3_acc_va:
                    best_worst3_acc_va = result['worst3_acc_va']
                    best_worst3_acc_te = result['worst3_acc_te']
            if args.SMA.K**2 >= 5:
                if result["worst5_grp_acc_va"] >= best_worst5_grp_acc_va:
                    best_worst5_grp_acc_va = result['worst5_grp_acc_va']
                    best_worst5_grp_acc_te = result['worst5_grp_acc_te']
                
                
        # model.save(checkpoint_file) # Deactivate saving model for now
        # result["args"] = OmegaConf.to_container(result["args"], resolve=True)
        # print(json.dumps(result))
        log_dict = results_to_log_dict(result, args.SMA.mode)

        wandb.log(log_dict, step=epoch)
    # log in the end the best acc as summary
    wandb.run.summary["best_mean_group_acc_va"] = best_mean_group_acc_va
    wandb.run.summary["best_mean_group_acc_te"] = best_mean_group_acc_te
    wandb.run.summary["best_acc_va"] = best_acc_va
    wandb.run.summary["best_acc_te"] = best_acc_te
    wandb.run.summary["best_worst_acc_va"] = best_worst_acc_va
    wandb.run.summary["best_worst_acc_te"] = best_worst_acc_te
    if args.SMA.K**2 >= 3:
        wandb.run.summary["best_worst3_acc_va"] = best_worst3_acc_va
        wandb.run.summary["best_worst3_acc_te"] = best_worst3_acc_te
    if args.SMA.K**2 >= 5:
        wandb.run.summary["best_worst5_grp_acc_va"] = best_worst5_grp_acc_va
        wandb.run.summary["best_worst5_grp_acc_te"] = best_worst5_grp_acc_te
        
    wandb.finish()
    return best_mean_group_acc_va


def get_ray_evaluator(run_function):
    # Default arguments for Ray: 1 worker and 1 worker per evaluation
    method_kwargs = {
        "num_cpus": 1,
        "num_cpus_per_task": 1,
        "callbacks": [TqdmCallback()]
    }

    # If GPU devices are detected then it will create 'n_gpus' workers
    # and use 1 worker for each evaluation

    method_kwargs["num_cpus"] = 1
    method_kwargs["num_gpus"] = 1
    method_kwargs["num_cpus_per_task"] = 1
    method_kwargs["num_gpus_per_task"] = 1

    evaluator = Evaluator.create(
        run_function,
        method="ray",
        method_kwargs=method_kwargs
    )
    print(
        f"Created new evaluator with {evaluator.num_workers} worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}", )

    return evaluator


if __name__ == "__main__":

    args_command_line = parse_args()
    config_dict = OmegaConf.to_container(
        OmegaConf.load(os.path.join('configs',
                                    args_command_line.config)))
    args = OmegaConf.create(config_dict)
    args["config_name"] = args_command_line.config
    args["n_gpus"] = torch.cuda.device_count()
    # if args.SMA.mode is not defined, we set it to 'classic':
    args.SMA.mode = 'classic' if ('SMA' in args) and ('mode' not in args.SMA) else args.SMA.mode
    # 'medical-leaf', 'texture-dtd', '73sports', 'resisc', 'dogs'
    for dataset_name in args.SMA.names:
        args.SMA.name = dataset_name
        # Define your search and execute it

        for K in args.SMA.Ks:
            args.SMA.K = K
            for mu in args.SMA.mus:

                if args.SMA.mode == 'binary':
                    args.SMA.mu = mu / (K - 1)
                else:
                    args.SMA.mu = mu
                args.IR = args.SMA.mu * (K - 1)
                # ['erm','jtt', 'suby', 'subg', 'rwy', 'rwg', 'dro']
                for method in args.SMA.methods:
                    args.method = method
                    for split_seed in args.SMA.split_seeds:
                        args.SMA.split_seed = split_seed

                        args.group = f"K={args.SMA.K}_{args.method}"
                        args.group_best = f"{args.SMA.name}_K={args.SMA.K}_{args.method}_mu={args.SMA.mu}_seed={args.SMA.split_seed}"
                        if "selector" in args.wandb_project :
                            #add the selector at the start of the string
                            args.group_best = f"{args.selector}_{args.group_best}"
                        ##### HBO PART
                        problem = HpProblem()
                        problem.add_hyperparameter((1e-4, 5e-3, "log-uniform"), "lr", default_value=1e-3)

                       
	                problem.add_hyperparameter((8, 32, "log-uniform"), "batch_size", default_value=16)
	                problem.add_hyperparameter((1e-4, 1.0, "log-uniform"), "weight_decay", default_value=1e-3)
                        # problem.add_hyperparameter((1, 60), "T", default_value=40)
                        if method == 'jtt':
                            problem.add_hyperparameter([1, 3, 5], "T", default_value=3)
                            problem.add_hyperparameter([4,20,50,100], "up", default_value=20)

                        evaluator = get_ray_evaluator(run)
                        hbo_log_dir = f"./outputs/hbo_logdir/{args.group_best}"
                        search = CBO(problem, evaluator, verbose=1, random_state=42, log_dir=hbo_log_dir)
                        print("Number of workers: ", evaluator.num_workers)
                        print(problem.default_configuration)
                        print(f"GPU available: {torch.cuda.is_available()}")
                        results = search.search(max_evals=args.n_HBO_runs)
                        # print(results['objective'])
                        # print(results)

                        i_max = results.objective.argmax()
                        best_config = results.iloc[i_max][:-3].to_dict()
                        best_config = {k[2:]: v for k, v in best_config.items() if k.startswith("p:")}

                        print(
                            f"The best configuration found by DeepHyper has an accuracy {results['objective'].iloc[i_max]:.3f}, \n"
                        )

                        for k, v in best_config.items():
                            args[k] = v
                            print(f"{k}: {v}")

                            # now we use the best hyper parameters to rerun the model with different init seed :
                        print(f" ####  NOW RUNNING THE BEST CONFIGURATION WITH {args.n_eval_init_seed} DIFFERENT SEEDS ####")
                        for i in range(args.n_eval_init_seed)[
                                ::-1]:  # -1 to have reverse order to 0 in the end for next outer loop
                            args["init_seed"] = i
                            run()

            # print(json.dumps(best_config, indent=4))
