import sys
import sys
sys.path.append('../../ADATIME/')
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, F1Score
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections

from torchmetrics import Accuracy, AUROC, F1Score
from dataloader.dataloader import data_generator, few_shot_data_generator
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class
from configs.sweep_params import sweep_alg_hparams
from utils import fix_randomness, starting_logs, DictAsObject,AverageMeter
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

class AbstractTrainer(object):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device

        # Exp Description
        self.experiment_description = args.dataset 
        self.run_description = f"{args.da_method}_{args.exp_name}"
        
        # paths
        self.home_path =  os.getcwd() #os.path.dirname(os.getcwd())
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        # self.create_save_dir(os.path.join(self.home_path,  self.save_dir ))
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description, f"{self.run_description}")
        os.makedirs(self.exp_log_dir, exist_ok=True)




        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()

        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels
        self.dataset_configs.da_method = self.da_method
        if self.da_method in ['OSBP', 'DeepJDOT_BP']:
            self.dataset_configs.num_classes += 1
        print("src class : ", self.dataset_configs.num_classes)
        # Specify number of hparams
        self.hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}

        # metrics
        self.num_classes = self.dataset_configs.num_classes # + self.dataset_configs.n_src_private
        self.ACC = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.F1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.AUROC = AUROC(task="multiclass", num_classes=self.num_classes)        

        # metrics

    def sweep(self):
        # sweep configurations
        pass
    
    def initialize_algorithm(self):
        # get algorithm class
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

    def load_checkpoint(self, model_dir):
        checkpoint = torch.load(os.path.join(self.home_path, model_dir, 'checkpoint.pt'))
        last_model = checkpoint['last']
        best_model = checkpoint['best']
        return last_model, best_model

    def train_model(self):
        # Get the algorithm and the backbone network
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)

        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

        # Training the model
        self.last_model, self.best_model = self.algorithm.update(self.src_train_dl, self.trg_train_dl, self.loss_avg_meters, self.logger)
        return self.last_model, self.best_model
    
    def evaluate(self, test_loader):
        self.loss, self.full_logits, self.full_labels, self.full_preds = self.algorithm.evaluate(test_loader)

    def H_score(self, trg_pred, trg_y):
        class_c = np.where(trg_y != -1)
        class_p = np.where(trg_y == -1)
        print(np.array(class_p).shape)
        print(np.array(class_c).shape)


        #trg_pred = self.algorithm.decision_function(trg_pred)
        print(trg_pred.unique())
        print(trg_y.unique())
        label_c, pred_c = trg_y[class_c], trg_pred[class_c]
        label_p, pred_p = trg_y[class_p], trg_pred[class_p]
        acc_c = (pred_c == label_c).sum()/(len(pred_c)) if len(pred_c) != 0 else torch.Tensor([0])
        #acc_c = self.ACC(pred_c.argmax(dim=1), label_c)


        #pred_p = self.algorithm.decision_function(pred_p)
        acc_p = (pred_p == label_p).sum()/(len(pred_p)) if len(pred_p) != 0 else torch.Tensor([0])
        #acc_p = self.ACC(pred_p, label_p)

        acc_mix = (trg_y != -1).sum()/len(trg_y) * acc_c + (trg_y == -1).sum()/len(trg_y) * acc_p
        print("Trg Private Acc : ", acc_p.item())
        if acc_c == 0 or acc_p == 0:
            H = torch.Tensor([0])
        else:
            H = 2 * acc_c * acc_p / (acc_p + acc_c)
        return H, acc_c, acc_p, acc_mix

    def get_configs(self):
        dataset_class = get_dataset_class("SMA")
        hparams_class = get_hparams_class("SMA")
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id):
        self.src_train_dl, self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, True)

        self.trg_train_dl, self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, False)

        '''self.few_shot_dl_5 = few_shot_data_generator(self.trg_test_dl, self.dataset_configs,
                                                     5)  # set 5 to other value if you want other k-shot FST'''

    def create_save_dir(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def calculate_metrics_risks(self):
        # calculation based source test data
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        # calculation based few_shot test data
        #self.evaluate(self.few_shot_dl_5)
        #fst_risk = self.loss.item()
        # calculation based target test data
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        # calculate metrics
        acc = self.ACC(self.full_preds.cpu(), self.full_labels.cpu()).item()
        # f1_torch
        f1 = self.F1(self.full_preds.cpu(), self.full_labels.cpu()).item()
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()
        # f1_sk learn
        # f1 = f1_score(self.full_preds.argmax(dim=1).cpu().numpy(), self.full_labels.cpu().numpy(), average='macro')

        #risks = src_risk, fst_risk, trg_risk
        risks = src_risk, trg_risk
        metrics = acc, f1, auroc

        return risks, metrics

    def save_tables_to_file(self,table_results, name):
        # save to file if needed
        table_results.to_csv(os.path.join(self.exp_log_dir,f"{name}.csv"))

    def save_checkpoint(self, home_path, log_dir, last_model, best_model):
        save_dict = {
            "last": last_model,
            "best": best_model
        }
        # save classification report
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path)

    def calculate_avg_std_wandb_table(self, results):

        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}

        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        return results, summary_metrics

    def log_summary_metrics_wandb(self, results, risks):
       
        # Calculate average and standard deviation for metrics
        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]

        avg_risks = [np.mean(risks.get_column(risk)) for risk in risks.columns[2:]]
        std_risks = [np.std(risks.get_column(risk)) for risk in risks.columns[2:]]

        # Estimate summary metrics
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}
        summary_risks = {risk: np.mean(risks.get_column(risk)) for risk in risks.columns[2:]}


        # append avg and std values to metrics
        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        # append avg and std values to risks 
        results.add_data('mean', '-', *avg_risks)
        risks.add_data('std', '-', *std_risks)

    def wandb_logging(self, total_results, total_risks, summary_metrics, summary_risks):
        # log wandb
        wandb.log({'results': total_results})
        wandb.log({'risks': total_risks})
        wandb.log({'hparams': wandb.Table(dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']), allow_mixed_types=True)})
        wandb.log(summary_metrics)
        wandb.log(summary_risks)

    '''def calculate_metrics(self):
       
        self.evaluate(self.trg_test_dl)
        # accuracy  
        acc = self.ACC(self.full_preds.cpu(), self.full_labels.cpu()).item()
        # f1
        f1 = self.F1(self.full_preds.cpu(), self.full_labels.cpu()).item()
        # auroc 
        auroc = self.AUROC(self.full_logits.cpu(), self.full_labels.cpu()).item()

        print("Accuracy : ", acc)
        print("F1 : ", f1)
        print("AUROC : ", auroc)
        return acc, f1, auroc'''

    def calculate_metrics(self):
        self.evaluate(self.trg_test_dl)

        if self.dataset_configs.n_trg_private > 0:
            print("trg_private_class ::: ", self.trg_test_dl.dataset.trg_private)
            #mask = np.isin(self.full_labels.cpu(), self.trg_private_class, invert=True)
            mask = self.full_labels.cpu() == -1

            # accuracy
            #acc = (self.full_preds.argmax(dim=1).cpu() == self.full_labels.cpu()).numpy().mean()
            # acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()

            # f1
            #f1 = self.F1(self.full_preds[mask].argmax(dim=1).cpu(), self.full_labels[mask].cpu()).item()
            # auroc
            #auroc = 0  # self.AUROC(self.full_preds[mask].cpu(), self.full_labels[mask].cpu()).item()

            print("Before mask ", self.full_labels.unique())
            self.full_labels[mask] = -1
            print("self.dataset_configs.num_classes-1 : ", self.dataset_configs.num_classes-1)
            self.full_labels[self.full_labels>=self.dataset_configs.num_classes-1]
            print("total private : ", (self.full_labels == -1).sum())
            print("After mask ", self.full_labels.unique())
            H_score, acc_c, acc_p, acc_mix = self.H_score(self.full_preds.cpu(), self.full_labels.cpu())
            H_score, acc_c, acc_p, acc_mix = H_score.item(), acc_c.item(), acc_p.item(), acc_mix.item()
            #acc = (self.algorithm.decision_function(self.full_preds).cpu() == self.full_labels.cpu()).numpy().mean()
            print("H_score : ", H_score)
            print("Acc_C : ", acc_c)
            print("Acc_P : ", acc_p)
            print("Acc_Mix : ", acc_mix)

            return H_score, acc_c, acc_p, acc_mix

        # accuracy
        print("full_preds : ", self.full_preds, self.full_preds.shape)
        print("full_labels : ", self.full_labels, self.full_labels.shape)
        acc = self.ACC(self.full_preds.cpu(), self.full_labels.cpu()).item()
        # f1
        f1 = self.F1(self.full_preds.cpu(), self.full_labels.cpu()).item()
        # auroc
        auroc = self.AUROC(self.full_logits.cpu(), self.full_labels.cpu()).item()
        print("Acc : ", acc)
        print("F1 : ", f1)
        print("AUROC : ", auroc)
        return acc, f1, auroc

    def calculate_risks(self):
         # calculation based source test data
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        # calculation based few_shot test data
        #self.evaluate(self.few_shot_dl_5)
        #fst_risk = self.loss.item()
        # calculation based target test data
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        #return src_risk, fst_risk, trg_risk
        return src_risk, trg_risk

    def append_results_to_tables(self, table, scenario, run_id, metrics):

        # Create metrics and risks rows
        results_row = [scenario, run_id, *metrics]

        # Create new dataframes for each row
        results_df = pd.DataFrame([results_row], columns=table.columns)

        # Concatenate new dataframes with original dataframes
        table = pd.concat([table, results_df], ignore_index=True)

        return table
    
    def add_mean_std_table(self, table, columns):
        # Calculate average and standard deviation for metrics
        avg_metrics = [table[metric].mean() for metric in columns[2:]]
        std_metrics = [table[metric].std() for metric in columns[2:]]

        # Create dataframes for mean and std values
        mean_metrics_df = pd.DataFrame([['mean', '-', *avg_metrics]], columns=columns)
        std_metrics_df = pd.DataFrame([['std', '-', *std_metrics]], columns=columns)

        # Concatenate original dataframes with mean and std dataframes
        table = pd.concat([table, mean_metrics_df, std_metrics_df], ignore_index=True)

        # Create a formatting function to format each element in the tables
        format_func = lambda x: f"{x:.4f}" if isinstance(x, float) else x

        # Apply the formatting function to each element in the tables
        table = table.applymap(format_func)

        return table 