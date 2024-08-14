import itertools
from copy import deepcopy

import numpy as np
import ot
import torch
from torch import nn
from torch.nn import BCELoss
import torch.nn.functional as F

from models.loss import Entropy, ConditionalEntropyLoss
from models.models import CLS, ReverseLayerF, ProtoCLS, MemoryQueue, LinearAverage, classifierOVANet, DiscriminatorUDA
from utils import sinkhorn, adaptive_filling, ubot_CCD


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs, backbone):
        super(Algorithm, self).__init__()
        self.configs = configs

        self.cross_entropy = nn.CrossEntropyLoss()
        self.feature_extractor = backbone(configs)
        self.classifier = CLS(configs)  # classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

    # update function is common to all algorithms
    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    # train loop vary from one method to another
    def training_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, test_loader):
        feature_extractor = self.feature_extractor.to(self.device)
        classifier = self.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, logits_list, labels_list, preds_list = [], [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features)

                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.item())
                logits = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability
                preds = logits.argmax(axis=1)

                # append predictions and labels
                logits_list.append(logits)
                labels_list.append(labels)
                preds_list.append(preds)

        loss = torch.tensor(total_loss).mean()  # average loss
        full_logits = torch.cat((logits_list))
        full_labels = torch.cat((labels_list))
        full_preds = torch.cat((preds_list))
        return loss, full_logits, full_labels, full_preds


class OSBP(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)


        # optimizer and scheduler

        #self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device


        self.optimizer = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_disc = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.t = hparams["t"]

        #self.bce = BCEWithLogitsLoss()
        self.bce = BCELoss()
        self.is_uniDA = True

    def my_bce(self, p):
        return torch.log(p) - torch.log(1-p)
    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        '''nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs+1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]') #TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')'''

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop
            #self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model
    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            src_cls_loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()


            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()
            trg_feat = self.feature_extractor(trg_x)
            rev_trg_feat = ReverseLayerF.apply(trg_feat, alpha)
            trg_pred = self.classifier(rev_trg_feat)
            trg_soft = F.softmax(trg_pred)

            prob1 = torch.sum(trg_soft[:, :- 1], 1).view(-1, 1)
            #prob2 = trg_soft[:, - 1].contiguous().view(-1, 1)
            prob2 = trg_soft[:, - 1].view(-1, 1)


            target_funk = torch.FloatTensor(trg_pred.size()[0]).fill_(0.5).cuda()
            #target_funk = torch.FloatTensor(trg_pred.size()[0], 2).fill_(0.5).cuda()

            #prob = F.softmax(torch.cat((prob1, prob2), 1))
            prob = torch.cat((prob1, prob2), 1)


            #print(prob.shape, target_funk.shape, prob.max(), prob.min())
            #loss_t = self.bce(prob1.squeeze(), target_funk)
            loss_t = self.bce(prob2.squeeze(), target_funk)
            #loss_t = self.my_bce(prob2.squeeze())
            #loss_t = self.mybce(prob2.squeeze())
            #print(trg_pred.shape)


            loss_t.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses = {'Total_loss': src_cls_loss.item() + loss_t.item(), 'Src_cls_loss': src_cls_loss.item(), 'Adv Loss' : loss_t.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def evaluate(self, test_loader):
        feature_extractor = self.feature_extractor.to(self.device)
        classifier = self.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, logits_list, labels_list, preds_list = [], [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features)

                #Concat all private targets to same class
                mask = labels >= predictions.shape[-1]-1
                labels[mask] = predictions.shape[-1]-1

                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.detach().cpu().item())
                #predictions = self.algorithm.correct_predictions(predictions)
                logits = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability
                preds = predictions.argmax(dim=1)
                labels[mask] = -1
                mask = preds>=logits.shape[-1]-1
                preds[mask] = -1

                # append predictions and labels
                logits_list.append(logits)
                labels_list.append(labels)
                preds_list.append(preds)
        loss = torch.tensor(total_loss).mean()  # average loss
        full_logits = torch.cat((logits_list))
        full_labels = torch.cat((labels_list))
        full_preds = torch.cat((preds_list))
        return loss, full_logits, full_labels, full_preds


class UniOT(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        print(configs)
        # device
        self.device = device
        self.feature_extractor = backbone(configs).to(self.device)
        self.classifier = CLS(configs, uniOT=True).to(self.device)
        self.cluster_head = ProtoCLS(configs.feat_dim, hparams['K']).to(self.device)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # hparams
        self.hparams = hparams
        self.nb_classes = configs.num_classes


        # initialize the gamma (coupling in OT) with zeros
        self.gamma = torch.zeros(hparams["batch_size"],
                                 hparams["batch_size"])  # .dnn.K.zeros(shape=(self.batch_size, self.batch_size))
        self.gamma.to(self.device)


        self.optimizer = torch.optim.Adam(
            list(self.feature_extractor.parameters())+list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_feat = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_cls = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_cluhead = torch.optim.Adam(
            self.cluster_head.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.n_batch = int(hparams['MQ_size']/hparams['batch_size'])
        feat_dim = configs.feat_dim
        self.memqueue = MemoryQueue(feat_dim, hparams['batch_size'], self.n_batch, hparams['temp']).cuda()
        self.beta = None
        self.softmax = torch.nn.Softmax(dim=1)
        self.bce = BCELoss()
        self.is_uniDA = True
        self.t = 0.5


    def init_queue(self, dataloader):
        cnt_i = 0
        while cnt_i < self.n_batch:
            for x,y, id in dataloader:
                x, y, id = x.to(self.device), y.to(self.device), id.to(self.device)
                feats = self.feature_extractor(x)
                proto, preds = self.classifier(feats)
                self.memqueue.update_queue(F.normalize(proto), id)
                cnt_i += 1
                if cnt_i > self.n_batch - 1:
                    break

    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        '''nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')  # TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')'''

        self.init_queue(trg_loader)
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop
            # self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y, _ in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            _, src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)


    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):

        # Construct Joint Loaders
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))
        temp = self.hparams['temp']
        #soft = nn.Softmax(dim=1)
        for step, ((src_x, src_y, id_source), (trg_x, _, id_target)) in joint_loader:
            """if src_x.shape[0] != trg_x.shape[0]:
                continue"""

            if src_x.shape[0] > trg_x.shape[0]:
                src_x = src_x[:trg_x.shape[0]]
                src_y = src_y[:trg_x.shape[0]]
            elif trg_x.shape[0] > src_x.shape[0]:
                trg_x = trg_x[:src_x.shape[0]]

            batch_size = len(src_x)
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(
                self.device)  # extract source features

            #feature_ex_s = self.feature_extractor(src_x)
            #feature_ex_t = self.feature_extractor(trg_x)

            before_lincls_feat_s, after_lincls_s = self.classifier(self.feature_extractor(src_x))
            before_lincls_feat_t, after_lincls_t = self.classifier(self.feature_extractor(trg_x))

            #norm_feat_s = F.normalize(before_lincls_feat_s)
            norm_feat_t = F.normalize(before_lincls_feat_t)

            after_cluhead_t = self.cluster_head(before_lincls_feat_t)

            # =====Source Supervision=====
            criterion = nn.CrossEntropyLoss().cuda()
            loss_cls = criterion(after_lincls_s, src_y)

            # =====Private Class Discovery=====
            minibatch_size = norm_feat_t.size(0)

            # obtain nearest neighbor from memory queue and current mini-batch
            feat_mat2 = torch.matmul(norm_feat_t, norm_feat_t.t()) / temp
            mask = torch.eye(feat_mat2.size(0), feat_mat2.size(0)).bool().cuda()
            feat_mat2.masked_fill_(mask, -1 / temp)

            nb_value_tt, nb_feat_tt = self.memqueue.get_nearest_neighbor(norm_feat_t, id_target.cuda())
            neighbor_candidate_sim = torch.cat([nb_value_tt.reshape(-1, 1), feat_mat2], 1)
            values, indices = torch.max(neighbor_candidate_sim, 1)
            neighbor_norm_feat = torch.zeros((minibatch_size, norm_feat_t.shape[1])).cuda()
            for i in range(minibatch_size):
                neighbor_candidate_feat = torch.cat([nb_feat_tt[i].reshape(1, -1), norm_feat_t], 0)
                neighbor_norm_feat[i, :] = neighbor_candidate_feat[indices[i], :]

            neighbor_output = self.cluster_head(neighbor_norm_feat)

            # fill input features with memory queue
            fill_size_ot = self.hparams['K']
            mqfill_feat_t = self.memqueue.random_sample(fill_size_ot)
            mqfill_output_t = self.cluster_head(mqfill_feat_t)

            # OT process
            # mini-batch feat (anchor) | neighbor feat | filled feat (sampled from memory queue)
            S_tt = torch.cat([after_cluhead_t, neighbor_output, mqfill_output_t], 0)
            #print(mqfill_output_t.shape, after_cluhead_t.shape, neighbor_output.shape, S_tt.shape)
            S_tt *= temp
            Q_tt = sinkhorn(S_tt.detach(), epsilon=0.05, sinkhorn_iterations=3)
            Q_tt_tilde = Q_tt * Q_tt.size(0)
            anchor_Q = Q_tt_tilde[:minibatch_size, :]
            neighbor_Q = Q_tt_tilde[minibatch_size:2 * minibatch_size, :]

            # compute loss_PCD
            loss_local = 0
            for i in range(minibatch_size):
                sub_loss_local = 0
                sub_loss_local += -torch.sum(neighbor_Q[i, :] * F.log_softmax(after_cluhead_t[i, :]))
                sub_loss_local += -torch.sum(anchor_Q[i, :] * F.log_softmax(neighbor_output[i, :]))
                sub_loss_local /= 2
                loss_local += sub_loss_local
            loss_local /= minibatch_size
            loss_global = -torch.mean(torch.sum(anchor_Q * F.log_softmax(after_cluhead_t, dim=1), dim=1))
            loss_PCD = (loss_global + loss_local) / 2

            # =====Common Class Detection=====
            #if global_step > 100:
            source_prototype = self.classifier.ProtoCLS.fc.weight
            if self.beta is None:
                self.beta = ot.unif(source_prototype.size()[0])

            # fill input features with memory queue
            fill_size_uot = self.n_batch * batch_size
            mqfill_feat_t = self.memqueue.random_sample(fill_size_uot)
            ubot_feature_t = torch.cat([mqfill_feat_t, norm_feat_t], 0)
            #full_size = ubot_feature_t.size(0)

            # Adaptive filling
            newsim, fake_size = adaptive_filling(ubot_feature_t, source_prototype, self.hparams['gamma'], self.beta, fill_size_uot)
            #newsim = torch.matmul(ubot_feature_t, source_prototype.t())
            #fake_size = 0

            # UOT-based CCD
            high_conf_label_id, high_conf_label, _, new_beta = ubot_CCD(newsim, self.beta, fake_size=fake_size,
                                                                        fill_size=fill_size_uot, mode='minibatch')
            # adaptive update for marginal probability vector
            self.beta = self.hparams['mu'] * self.beta + (1 - self.hparams['mu']) * new_beta

            # fix the bug raised in https://github.com/changwxx/UniOT-for-UniDA/issues/1
            # Due to mini-batch sampling, current mini-batch samples might be all target-private.
            # (especially when target-private samples dominate target domain, e.g. OfficeHome)
            if high_conf_label_id.size(0) > 0:
                loss_CCD = criterion(after_lincls_t[high_conf_label_id, :], high_conf_label[high_conf_label_id])
            else:
                loss_CCD = 0

            loss_all = loss_cls + self.hparams['lam'] * (loss_CCD)

            self.optimizer_feat.zero_grad()
            self.optimizer_cls.zero_grad()
            self.optimizer_cluhead.zero_grad()
            loss_all.backward()
            self.optimizer_feat.step()
            self.optimizer_cls.step()
            self.optimizer_cluhead.step()

            self.classifier.ProtoCLS.weight_norm()  # very important for proto-classifier
            self.cluster_head.weight_norm()  # very important for proto-classifier
            self.memqueue.update_queue(norm_feat_t, id_target.cuda())

            losses = {'Total_loss': loss_all.item(), 'loss_cls': loss_cls.item(),
                      'loss_PCD': loss_PCD.item(),
                      'loss_CCD': loss_CCD}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def evaluate(self, test_loader, src=False):
        feature_extractor = self.feature_extractor.to(self.device)
        classifier = self.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, logits_list, labels_list, preds_list = [], [], [], []
        norm_feat_t_list = []

        with torch.no_grad():
            for data, labels, id in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = feature_extractor(data)
                before_lincls_feat_t, predictions = classifier(features)
                norm_feat_t = F.normalize(before_lincls_feat_t)

                #if test_loader.dataset.is_src:
                mask = labels < predictions.shape[-1]
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())

                #predictions = self.algorithm.correct_predictions(predictions)
                logits = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability
                preds_list.append(logits.argmax(dim=1))

                # append predictions and labels
                logits_list.append(logits.detach().cpu())
                labels[~mask] = -1
                labels_list.append(labels.detach().cpu())
                norm_feat_t_list.append(norm_feat_t.detach().cpu())
        loss = torch.tensor(total_loss).mean()  # average loss
        full_logits = torch.cat((logits_list))
        full_labels = torch.cat((labels_list))
        full_preds = torch.cat((preds_list))
        norm_feat_t = torch.cat((norm_feat_t_list))

        source_prototype = classifier.ProtoCLS.fc.weight

        stopThr = 1e-6
        # Adaptive filling
        newsim, fake_size = adaptive_filling(norm_feat_t.cuda(),
                                             source_prototype, self.hparams['gamma'], self.beta, 0, stopThr=stopThr)

        # obtain predict label
        _, __, pred_label, ___ = ubot_CCD(newsim, self.beta, fake_size=fake_size, fill_size=0, mode='minibatch',
                                          stopThr=stopThr)
        pred_label = pred_label.cpu().data.numpy()
        print("new labels : ", np.unique(pred_label))
        mask = pred_label == self.nb_classes
        full_preds[mask] = -1
        #full_preds[mask] *= 0

        return loss, full_logits, full_labels, full_preds

    def get_latent_features(self, dataloader):
        feature_set = []
        label_set = []
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (data, label, _) in enumerate(dataloader):
                data = data.to(self.device)
                _, feature = self.classifier(self.feature_extractor(data))
                feature_set.append(feature.cpu())
                label_set.append(label.cpu())
            feature_set = torch.cat(feature_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
        return feature_set, label_set

class DANCE(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler

        # self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device
        #self.classifier = classifierNoBias(configs)
        self.rho = np.log(self.configs.num_classes)/2.0

        self.optimizer_feature_gen = torch.optim.Adam(
            list(self.feature_extractor.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_clasifier = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.entropy = Entropy()
        self.hidden_size = configs.in_dim

    def init_memory(self, trg_loader):
        self.ndata = len(trg_loader.dataset.labels)
        '''for (trg_x, _, _) in trg_loader:
            self.ndata += len(trg_x)'''
        print(self.ndata)
        self.lemniscate = LinearAverage(self.hidden_size, self.ndata).to(self.device)

    def entropy(self, p):
        p = F.softmax(p)
        return -torch.mean(torch.sum(p * torch.log(p + 1e-5), 1))

    def entropy_margin(self, p, value, margin=0.2, weight=None):
        p = F.softmax(p)
        return -torch.mean(self.hinge(torch.abs(-torch.sum(p * torch.log(p + 1e-5), 1) - value), margin))

    def hinge(self, input, margin=0.2):
        return torch.clamp(input, min=margin)

    def update(self, src_loader, trg_loader, avg_meter, logger):
        self.init_memory(trg_loader)
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        '''nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs + 1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]')  # TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')'''

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop
            # self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_loader, avg_meter):

        for src_x, src_y, _ in src_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            loss = src_cls_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            losses = {'Pr_Src_cls_loss': loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)
    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        # Combine dataloaders
        # Method 1 (min len of both domains)
        # joint_loader = enumerate(zip(src_loader, trg_loader))

        # Method 2 (max len of both domains)
        # joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y, _), (trg_x, _, trg_index)) in joint_loader:

            src_x, src_y, trg_x, trg_index = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device), trg_index.to(self.device)

            # zero grad
            self.optimizer_clasifier.zero_grad()
            self.optimizer_feature_gen.zero_grad()

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)
            trg_feat = F.normalize(trg_feat)

            feat_mat = self.lemniscate(trg_feat, trg_index)
            feat_mat[:, trg_index] = -1.0
            ### Calculate mini-batch x mini-batch similarity

            feat_mat2 = torch.matmul(trg_feat, trg_feat.t())
            mask = torch.eye(feat_mat2.size(0), feat_mat2.size(0)).bool().to(self.device)
            feat_mat2.masked_fill_(mask, -1)

            loss_nc = self.hparams["eta"] * self.entropy(torch.cat([trg_pred, feat_mat,feat_mat2], 1))
            loss_ent = self.hparams["eta"] * self.entropy_margin(trg_pred, self.rho, self.hparams["margin"])
            total_loss = loss_nc + src_cls_loss + loss_ent

            total_loss.backward()
            self.optimizer_feature_gen.step()
            self.optimizer_clasifier.step()
            self.optimizer_feature_gen.zero_grad()
            self.optimizer_clasifier.zero_grad()

            self.lemniscate.update_weight(trg_feat, trg_index)

            losses = {'Total_loss': total_loss.item(), 'Ent Loss': loss_ent.item(),
                      'Src_cls_loss': src_cls_loss.item(),
                      "Neighbors Clustering ": loss_nc.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def evaluate(self, test_loader):
        feature_extractor = self.feature_extractor.to(self.device)
        classifier = self.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, logits_list, labels_list, preds_list = [], [], [], []

        with torch.no_grad():
            for data, labels, _ in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = self.feature_extractor(data)
                predictions = F.softmax(self.classifier(features))

                entr = -torch.sum(predictions * torch.log(predictions), 1).data.cpu().numpy()

                conf, preds = predictions.max(dim=1)

                pred_unk = np.where(entr > self.rho)
                preds[pred_unk] = -1
                #mask = labels >= predictions.shape[-1]

                mask = labels < predictions.shape[-1]
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())
                mask = labels >= predictions.shape[-1]
                labels[mask] = -1
                # predictions = self.algorithm.correct_predictions(predictions)
                logits = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                logits_list.append(logits)
                labels_list.append(labels)
                preds_list.append(preds)
        loss = torch.tensor(total_loss).mean()  # average loss
        full_logits = torch.cat((logits_list))
        full_labels = torch.cat((labels_list))
        full_preds = torch.cat((preds_list))
        return loss, full_logits, full_labels, full_preds

    def get_latent_features(self, dataloader):
        feature_set = []
        label_set = []
        self.feature_extractor.eval()
        self.classifier.eval()
        with torch.no_grad():
            for _, (data, label, _) in enumerate(dataloader):
                data = data.to(self.device)
                feature = self.classifier(self.feature_extractor(data))
                feature_set.append(feature.cpu())
                label_set.append(label.cpu())
            feature_set = torch.cat(feature_set, dim=0)
            feature_set = F.normalize(feature_set, p=2, dim=-1)
            label_set = torch.cat(label_set, dim=0)
        return feature_set, label_set



class OVANet(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)


        # optimizer and scheduler

        #self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Domain Discriminator
        self.open_set_classifier = classifierOVANet(configs)

        self.optimizer_feature_gen = torch.optim.Adam(
            list(self.feature_extractor.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_clasifier = torch.optim.Adam(
            list(self.open_set_classifier.parameters()) + list(self.classifier.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.entropy = Entropy()

    def ova_loss(self, open_preds, label):
        assert len(open_preds.size()) == 3
        assert open_preds.size(1) == 2

        out_open = F.softmax(open_preds, 1)
        label_p = torch.zeros((out_open.size(0),
                               out_open.size(2))).long().cuda()
        label_range = torch.range(0, out_open.size(0) - 1).long()
        label_p[label_range, label] = 1
        label_n = 1 - label_p
        open_loss_pos = torch.mean(torch.sum(-torch.log(out_open[:, 1, :]
                                                        + 1e-8) * label_p, 1))
        open_loss_neg = torch.mean(torch.max(-torch.log(out_open[:, 0, :] +
                                                        1e-8) * label_n, 1)[0])
        return open_loss_pos, open_loss_neg

    def open_entropy(self, open_preds):
        assert len(open_preds.size()) == 3
        assert open_preds.size(1) == 2
        out_open = F.softmax(open_preds, 1)
        ent_open = torch.mean(torch.mean(torch.sum(-out_open * torch.log(out_open + 1e-8), 1), 1))
        return ent_open

    def entropy(self, p, prob=True, mean=True):
        if prob:
            p = F.softmax(p)
        en = -torch.sum(p * torch.log(p + 1e-5), 1)
        if mean:
            return torch.mean(en)
        else:
            return en

    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        '''nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs+1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]') #TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')'''

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop
            #self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        # Combine dataloaders
        # Method 1 (min len of both domains)
        # joint_loader = enumerate(zip(src_loader, trg_loader))

        # Method 2 (max len of both domains)
        # joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer_clasifier.zero_grad()
            self.optimizer_feature_gen.zero_grad()

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)
            src_open = self.open_set_classifier(src_feat)
            src_open = src_open.view(src_open.size(0), 2, -1)

            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)
            open_loss_pos, open_loss_neg = self.ova_loss(src_open, src_y)
            ## b x 2 x C
            loss_open = 0.5 * (open_loss_pos + open_loss_neg)
            total_loss = loss_open + src_cls_loss

            trg_feat = self.feature_extractor(trg_x)
            trg_open_pred = self.open_set_classifier(trg_feat)
            trg_open_pred = trg_open_pred.view(trg_open_pred.size(0), 2, -1)

            out_open_t = trg_open_pred.view(trg_x.size(0), 2, -1)
            ent_open = self.open_entropy(out_open_t)
            total_loss += ent_open

            total_loss.backward()
            self.optimizer_feature_gen.step()
            self.optimizer_clasifier.step()
            self.optimizer_feature_gen.zero_grad()
            self.optimizer_clasifier.zero_grad()

            losses = {'Total_loss': total_loss.item(), 'Open Loss': loss_open.item(), 'Src_cls_loss': src_cls_loss.item(),
                      "Entropy Open": ent_open.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def evaluate(self, test_loader):
        feature_extractor = self.feature_extractor.to(self.device)
        classifier = self.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, logits_list, labels_list, preds_list = [], [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = self.feature_extractor(data)
                predictions = F.softmax(self.classifier(features))
                open_preds = self.open_set_classifier(features)

                # open_class = len(predictions.shape[0])

                conf, preds = predictions.max(dim=1)
                # entr = -1*torch.sum(predictions*torch.log(predictions), 1).data.cpu().numpy()

                open_preds = F.softmax(open_preds.view(predictions.size(0), 2, -1), 1)
                tmp_range = torch.range(0, predictions.size(0) - 1).long().cuda()
                pred_unk = open_preds[tmp_range, 0, preds]
                ind_unk = np.where(pred_unk.data.cpu().numpy() > 0.5)[0]
                preds[ind_unk] = -1
                preds_list.append(preds)
                #mask = labels >= predictions.shape[-1]

                mask = labels < predictions.shape[-1]
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())
                mask = labels >= predictions.shape[-1]
                labels[mask] = -1
                # predictions = self.algorithm.correct_predictions(predictions)
                logits = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability

                # append predictions and labels
                logits_list.append(logits)
                labels_list.append(labels)
        loss = torch.tensor(total_loss).mean()  # average loss
        full_logits = torch.cat((logits_list))
        full_preds = torch.cat((preds_list))
        full_labels = torch.cat((labels_list))
        print("full preds : ", full_preds)
        return loss, full_logits, full_labels, full_preds

class UDA(Algorithm):
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)


        # optimizer and scheduler

        #self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        # hparams
        self.hparams = hparams
        # device
        self.device = device

        # Domain Discriminator
        self.domain_classifier = DiscriminatorUDA(configs)
        self.adv_discriminator = DiscriminatorUDA(configs)

        self.conditional_entropy = ConditionalEntropyLoss()
        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.adv_discriminator.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.w_0 = hparams["w0"]

        #self.bce = BCEWithLogitsLoss()
        self.bce = BCELoss()

    def normalize_weight(self, x):
        min_val = x.min()
        max_val = x.max()
        x = (x - min_val) / (max_val - min_val)
        #print(torch.mean(x))
        x = x/torch.mean(x)
        #assert (x > 1.0).any() or (x < 0.0).any()
        #x = (x-torch.mean(x))/torch.std(x)
        return x.detach()

    def reverse_sigmoid(self, y):
        return torch.log(y / (1.0 - y + 1e-10) + 1e-10)
    def get_src_weights(self, domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
        before_softmax = before_softmax / class_temperature
        after_softmax = nn.Softmax(-1)(before_softmax)
        domain_logit = self.reverse_sigmoid(domain_out)
        domain_logit = domain_logit / domain_temperature
        domain_out = nn.Sigmoid()(domain_logit)

        entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
        entropy_norm = entropy / np.log(after_softmax.size(1))
        #entropy_norm = self.normalize_weight(entropy_norm)
        #assert (entropy_norm > 1.0).any() or (entropy_norm < 0.0).any() == False
        weight = entropy_norm - domain_out
        #print(max(weight))
        weight = weight.detach()
        return weight
    def get_trg_weights(self, domain_out, before_softmax, domain_temperature=1.0, class_temperature=1.0):
        return -1*self.get_src_weights(domain_out, before_softmax, domain_temperature, class_temperature)

    def update(self, src_loader, trg_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        '''nb_pr_epochs = self.hparams["num_epochs_pr"]
        for epoch in range(1, nb_pr_epochs+1):
            self.pretrain_epoch(src_loader, avg_meter)

            logger.debug(f'[Pr Epoch : {epoch}/{nb_pr_epochs}]') #TODO : self.hparams["num_pr_epochs"]
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')'''

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # source pretraining loop
            #self.pretrain_epoch(src_loader, avg_meter)

            # training loop
            self.training_epoch(src_loader, trg_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        last_model = self.network.state_dict()

        return last_model, best_model

    def training_epoch(self, src_loader, trg_loader, avg_meter, epoch):
        # Combine dataloaders
        # Method 1 (min len of both domains)
        # joint_loader = enumerate(zip(src_loader, trg_loader))

        # Method 2 (max len of both domains)
        # joint_loader =enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        joint_loader = enumerate(zip(src_loader, itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:

            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)

            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # zero grad
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            trg_feat = self.feature_extractor(trg_x)
            trg_pred = self.classifier(trg_feat)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # Adv Domain Discriminator loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_adv_pred = self.adv_discriminator(src_feat_reversed)

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_adv_pred = self.adv_discriminator(trg_feat_reversed)

            # Domain classifier and weights computation
            src_domain_pred = self.domain_classifier(src_feat)
            trg_domain_pred = self.domain_classifier(trg_feat)

            src_temp = 10
            '''w_s = self.normalize_weight(self.conditional_entropy(src_domain_pred/src_temp)/np.log(len(src_domain_pred)) - src_domain_pred/src_temp)
            w_t = self.normalize_weight(trg_domain_pred - self.conditional_entropy(trg_domain_pred)/np.log(len(trg_domain_pred)))'''
            w_s = self.normalize_weight(self.get_src_weights(src_domain_pred, src_pred))
            w_t = self.normalize_weight(self.get_trg_weights(trg_domain_pred, trg_pred))
            # print(w_t)

            src_domain_loss = self.bce(src_domain_pred.squeeze(), domain_label_src)
            trg_domain_loss = self.bce(trg_domain_pred.squeeze(), domain_label_trg)

            '''src_adv_loss = w_s * F.cross_entropy(src_adv_pred, domain_label_src.long(), reduction='none')
            src_adv_loss = src_adv_loss.mean()
            trg_adv_loss = w_t * F.cross_entropy(trg_adv_pred, domain_label_trg.long(), reduction='none')
            trg_adv_loss = trg_adv_loss.mean()'''

            src_adv_loss = w_s * F.binary_cross_entropy(src_adv_pred.squeeze(), domain_label_src, reduction='none')
            src_adv_loss = src_adv_loss.mean()
            trg_adv_loss = w_t * F.binary_cross_entropy(trg_adv_pred.squeeze(), domain_label_trg, reduction='none')
            trg_adv_loss = trg_adv_loss.mean()

            # Task classification  Loss
            """mask = w_s < self.w_0/2
            print("Wrong Src : ", mask.sum())
            w_s2 = w_s.clone()
            w_s2[mask] = 0
            src_cls_loss = (w_s2)*F.cross_entropy(src_pred.squeeze(), src_y, reduction='none')
            src_cls_loss = src_cls_loss.mean()"""

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss
            adv_loss = src_adv_loss + trg_adv_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                   self.hparams["domain_loss_wt"] * adv_loss

            loss.backward(retain_graph=True)
            domain_loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            losses = {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                      "Adv Loss": adv_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

    def evaluate(self, test_loader):
        self.feature_extractor.eval()
        self.classifier.eval()

        total_loss, logits_list, labels_list, preds_list = [], [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)

                # forward pass
                features = self.feature_extractor(data)
                predictions = self.classifier(features)
                trg_domain_pred = self.domain_classifier(features)
                #w_t = trg_domain_pred - self.conditional_entropy(trg_domain_pred)/np.log(len(trg_domain_pred))
                w_t = self.normalize_weight(self.get_trg_weights(trg_domain_pred, predictions))
                #print(w_t)
                mask = w_t < self.w_0
                conf, preds = predictions.max(dim=1)
                preds[mask.squeeze()] = -1
                preds_list.append(preds)

                mask = labels < predictions.shape[-1]
                loss = F.cross_entropy(predictions[mask], labels[mask])
                total_loss.append(loss.detach().cpu().item())
                # predictions = self.algorithm.correct_predictions(predictions)
                logits = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability
                mask = labels >= predictions.shape[-1]
                labels[mask] = -1

                # append predictions and labels
                logits_list.append(logits)
                labels_list.append(labels)

        loss = torch.tensor(total_loss).mean()  # average loss
        full_logits = torch.cat((logits_list))
        full_labels = torch.cat((labels_list))
        full_preds = torch.cat((preds_list))
        return loss, full_logits, full_labels, full_preds

    def decision_function(self, preds):
        mask = preds.sum(axis=1) == 0.0
        confidence, pred = preds.max(dim=1)
        pred[mask] = -1
        return pred

