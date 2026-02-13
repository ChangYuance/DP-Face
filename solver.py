import torch
import torch.nn as nn
import time
import os
import seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from models import *
from datasets import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List, Tuple, Optional, Union, Dict, Any
import torch.nn.functional as F
from thop import profile
from pathlib import Path
from sklearn.metrics import roc_auc_score

class Solver(object):
    def __init__(self, args):
        """Init the global settings including device, seed, models, dataloaders, crterions, optimizers and schedulers
        Args:
            args
        """
        super(Solver, self).__init__()
        self.args = args
        self.log_path = os.path.join(self.args.output_path, "log.txt")
        # self.emotions = ["hap", "sad", "neu", "ang", "sur", "dis", "fea"]
        self.emotions = ["Normal", "Palsy"]
        self.best_wa = 0
        self.best_ua = 0
        self.best_test_wa = 0
        self.best_test_ua = 0
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)
        self.writer = SummaryWriter(log_dir=self.args.output_path)
        # init cuda
        if len(self.args.gpu_ids) > 0:
            torch.cuda.set_device(self.args.gpu_ids[0])
        self.device = torch.device(
            'cuda:%d' % self.args.gpu_ids[0] if self.args.gpu_ids else 'cpu')
        # set seed
        seed = self.args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        # init model
        self.model = create_model(self.args)
        if len(self.args.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, self.args.gpu_ids)
        self.model.to(self.device)
        # init dataloader
        self.train_dataloader = create_dataloader(self.args, "train")
        self.test_dataloader = create_dataloader(self.args, "test")
        if self.args.val:
            self.val_dataloader = create_dataloader(self.args, "val")
        # init criterion
        if self.args.focal_loss:
            self.criterion = FocalLoss(alpha=0.25, gamma=2, num_classes=self.args.num_classes).to(self.device)
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.args.label_smoothing).to(self.device)
        # init optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                        lr=self.args.lr,
                                        eps=self.args.eps,
                                        weight_decay=self.args.weight_decay)
        self.scheduler = build_scheduler(
            self.args, self.optimizer, len(self.train_dataloader))
        # resume
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cuda:0',  weights_only=False)
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
            self.args.start_epoch = 1
            self.model.load_state_dict(checkpoint['state_dict'])

        dummy_images = torch.randn(1, self.args.num_frames, 3, 224, 224).to(self.device)
        if self.args.AUs:
            dummy_AUs = torch.randn(1, self.args.num_frames, 20).to(self.device)
            flops, params = profile(self.model, inputs=(dummy_images, dummy_AUs))
        elif self.args.audio:
            dummy_audio = torch.randn(1, 600, 128).to(self.device)
            flops, params = profile(self.model, inputs=(dummy_images, dummy_audio))
        else:
            flops, params = profile(self.model, inputs=(dummy_images,))
        print(f"Model Params: {params/1e6:.2f} M")
        print(f"Model FLOPs: {flops/1e9:.2f} G")
    def run(self):
        for epoch in range(self.args.start_epoch, self.args.epochs):
            inf = '********************' + str(epoch) + '********************'
            start_time = time.time()
            with open(self.log_path, 'a') as f:
                f.write(inf + '\n')
            print(inf)
            # train the model for one epoch
            if self.args.AUs or self.args.audio:
                train_acc, train_loss, train_center_loss = self.train_AUs(epoch)
            else:
                train_acc, train_loss, train_center_loss = self.train(epoch)
            lr = self.optimizer.param_groups[0]['lr']  # 获取当前的学习率
            self.writer.add_scalar('LR/LearningRate', lr, epoch)

            # validate the model
            mode = "val" if self.args.val else "test"
            if self.args.AUs or self.args.audio:
                val_acc, val_loss, val_center_loss, val_auc, val_wrong_0_to_1_paths, val_wrong_1_to_0_paths = self.validate_AUs(epoch, mode = mode)
            else:
                val_acc, val_loss, val_center_loss, val_auc, val_wrong_0_to_1_paths, val_wrong_1_to_0_paths = self.validate(epoch, mode = mode)
            l2_loss = compute_l2_loss(self.model, self.args.weight_decay)/len(self.train_dataloader)/self.args.batch_size
            # self.writer.add_scalar('Val/Loss', 0.7*val_loss+(1-val_acc[0])*0.3, epoch)
            # self.writer.add_scalar('Loss_Train/CE', train_loss, epoch)
            self.writer.add_scalar('Loss_Train/CenterLoss', train_center_loss, epoch)
            self.writer.add_scalar('Loss_Train/All', train_loss*0.2 + train_center_loss*0.5+0.3*l2_loss, epoch)
            # self.writer.add_scalar('Loss_Val/CE', val_loss, epoch)
            self.writer.add_scalar('Loss_Val/CenterLoss', val_center_loss, epoch)
            self.writer.add_scalar('Loss_Val/All', val_loss*0.2 + val_center_loss*0.5+0.3*l2_loss, epoch)
            self.writer.add_scalar('acc/Val_acc', val_acc[0], epoch)
            # remember best acc and save checkpoint
            is_best = (val_acc[0] > self.best_wa) or (
                val_acc[1] > self.best_ua)
            self.best_wa = max(val_acc[0], self.best_wa)
            self.best_ua = max(val_acc[1], self.best_ua)            # if True and self.args.val:
            if is_best and self.args.val:
                if self.args.AUs or self.args.audio:
                    test_acc, test_loss, _, auc, wrong_0_to_1_paths, wrong_1_to_0_paths  = self.validate_AUs(epoch, mode="test")
                else:
                    test_acc, test_loss, _, auc, wrong_0_to_1_paths, wrong_1_to_0_paths = self.validate(epoch, mode="test")
                self.best_test_wa = max(test_acc[0], self.best_test_wa)
                self.best_test_ua = max(test_acc[1], self.best_test_ua)
            self.writer.add_scalar('acc/Best_Test', self.best_test_wa, epoch)
            if self.args.val:
                self.save({'epoch': epoch,
                            'state_dict': self.model.state_dict(),
                            'best_wa': self.best_wa,
                            'best_ua': self.best_ua,
                            'best_test_wa': self.best_test_wa,
                            'best_test_ua': self.best_test_ua,
                            'optimizer': self.optimizer.state_dict(),
                            'args': self.args}, is_best)
            else:
                self.save({'epoch': epoch,
                            'state_dict': self.model.state_dict(),
                            'best_wa': self.best_wa,
                            'best_ua': self.best_ua,
                            'optimizer': self.optimizer.state_dict(),
                            'args': self.args}, is_best)
            # print and save log
            epoch_time = time.time() - start_time
            if self.args.val:
                msg = self.get_acc_msg_val(epoch, train_acc, train_loss,
                                    val_acc, val_loss,
                                    test_acc, test_loss,
                                    self.best_test_wa, self.best_test_ua, epoch_time)
            else:
                msg = self.get_acc_msg(epoch, train_acc, train_loss,
                                    val_acc, val_loss,
                                self.best_wa, self.best_ua, epoch_time)
            with open(self.log_path, 'a') as f:
                f.write(msg)
            print(msg)
            if epoch %5==0:
                cm_msg = self.get_confusion_msg(
                        val_acc[2],
                        val=False,
                        wrong_0_to_1_paths=val_wrong_0_to_1_paths,
                        wrong_1_to_0_paths=val_wrong_1_to_0_paths
                    )
                with open(self.log_path, 'a') as f:
                    f.write(cm_msg)
                print(cm_msg)
            if is_best:
                if self.args.val:
                    cm_msg_easy = self.get_confusion_msg_easy(
                        test_acc[2],
                        val=True,
                        auc = auc,
                        wrong_0_to_1_paths=wrong_0_to_1_paths,
                        wrong_1_to_0_paths=wrong_1_to_0_paths
                    )
                else:
                    cm_msg_easy = self.get_confusion_msg_easy(
                        val_acc[2],
                        val=False,
                        auc = auc,
                        wrong_0_to_1_paths=wrong_0_to_1_paths,
                        wrong_1_to_0_paths=wrong_1_to_0_paths
                    )

                with open(self.log_path, 'a') as f:
                    f.write(cm_msg_easy)
                print(cm_msg_easy)
                # convert confusion matrix to heatmap
                cm = []
                if self.args.val:
                    for row in test_acc[2]:
                        row = row / np.sum(row)
                        cm.append(row)
                else:
                    for row in val_acc[2]:
                        row = row / np.sum(row)
                        cm.append(row)
                fig_path = os.path.join(self.args.output_path, "fig_best.png")
                ax = seaborn.heatmap(
                    cm, xticklabels=self.emotions, yticklabels=self.emotions, cmap='rocket_r')
                figure = ax.get_figure()
                # save the heatmap
                figure.savefig(fig_path)
                plt.close()
        self.writer.close()
        return self.best_ua, self.best_ua

    def train(self, epoch):
        """ Train the model for one eopch
        """
        self.model.train()
        all_pred, all_target, allpath = [], [], []
        all_loss, all_center_loss = 0, 0
        for i, (images, pathname, target) in enumerate(self.train_dataloader):
            print("Training epoch \t{}: {}\\{}".format(
                epoch, i + 1, len(self.train_dataloader)), end='\r')
            images = images.to(self.device)
            target = target.to(self.device)
            center_loss = 0
            if self.args.center_loss:
                output, feature = self.model(images)
                center_loss = self.model.center_loss(feature, target)
            else:
                output = self.model(images)
            loss = self.criterion(output, target)
            pred = torch.argmax(output, 1).cpu().detach().numpy()
            target = target.cpu().numpy()
            all_pred.extend(pred)
            all_target.extend(target)
            all_loss += loss.item()
            if self.args.center_loss:
                all_center_loss += center_loss.item()
            self.optimizer.zero_grad()
            if self.args.center_loss:
                (0.3*loss + 0.7*center_loss).backward()
            else:
                loss.backward()
            self.optimizer.step()
            self.scheduler.step_update(epoch * len(self.train_dataloader) + i)
        # WAR
        acc1 = accuracy_score(all_target, all_pred)
        # UAR
        acc2 = balanced_accuracy_score(all_target, all_pred)
        loss = all_loss / len(self.train_dataloader)
        center_loss = all_center_loss / len(self.train_dataloader)
        return [acc1, acc2], loss, center_loss
    def train_AUs(self, epoch):
        """ Train the model for one eopch
        """
        self.model.train()
        all_pred, all_target, allpath = [], [], []
        all_loss, all_center_loss = 0, 0
        for i, (images, AUs, pathname, target) in enumerate(self.train_dataloader):
            print("Training epoch \t{}: {}\\{}".format(
                epoch, i + 1, len(self.train_dataloader)), end='\r')
            images = images.to(self.device)
            AUs = AUs.to(self.device)
            target = target.to(self.device)
            center_loss = 0
            if self.args.center_loss:
                output, feature = self.model(images, AUs)
                center_loss = self.model.center_loss(feature, target)
            else:
                output = self.model(images, AUs)
            # AUs.shape torch.Size([8, 16, 20])
            # images.shape torch.Size([8, 16, 3, 224, 224])
            loss = self.criterion(output, target)
            pred = torch.argmax(output, 1).cpu().detach().numpy()
            target = target.cpu().numpy()
            all_pred.extend(pred)
            all_target.extend(target)
            all_loss += loss.item()
            if self.args.center_loss:
                all_center_loss += center_loss.item()
            self.optimizer.zero_grad()
            if self.args.center_loss:
                (0.3*loss + 0.7*center_loss).backward()
            else:
                loss.backward()
            self.optimizer.step()
            self.scheduler.step_update(epoch * len(self.train_dataloader) + i)
        acc1 = accuracy_score(all_target, all_pred) # WAR
        acc2 = balanced_accuracy_score(all_target, all_pred) # UAR
        loss = all_loss / len(self.train_dataloader)
        center_loss = all_center_loss / len(self.train_dataloader)
        return [acc1, acc2], loss, center_loss
    def validate_AUs(self, epoch, mode="val"):
        """Validate the model for one epoch
        """
        self.model.eval()
        all_pred, all_target, allpath, all_output = [], [], [], []
        all_loss, all_center_loss = 0, 0
        if mode == "val":
            dataloader = self.val_dataloader
        elif mode == "test":
            dataloader = self.test_dataloader
        for i, (images, AUs, pathname, target) in enumerate(dataloader):
            print("Testing epoch \t{}: {}\\{}".format(
                epoch, i + 1, len(dataloader)), end='\r')
            images = images.to(self.device)
            AUs = AUs.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                if self.args.center_loss:
                    output, feature = self.model(images, AUs)
                    center_loss = self.model.center_loss(feature, target)
                else:
                    output = self.model(images, AUs)
            loss = self.criterion(output, target)
            pred = torch.argmax(output, 1).cpu().detach().numpy()
            target = target.cpu().numpy()
            all_pred.extend(pred)
            all_target.extend(target)
            all_loss += loss.item()
            if self.args.center_loss:
                all_center_loss += center_loss.item()
            all_output.append(output.cpu())
            allpath.extend(pathname)
        all_output = torch.cat(all_output, dim=0).cpu()
        probs = F.softmax(all_output, dim=1)[:, 1].numpy()  # 正类（class 1）概率
        auc = roc_auc_score(all_target, probs)
        acc1 = accuracy_score(all_target, all_pred)         # WAR
        acc2 = balanced_accuracy_score(all_target, all_pred)         # UAR
        c_m = confusion_matrix(all_target, all_pred)
        loss = all_loss / len(dataloader)
        center_loss = all_center_loss / len(dataloader)
        wrong_0_to_1_paths, wrong_1_to_0_paths = get_wrong_path_with_confidence(
            all_pred, all_target, allpath, all_output,
            top_k=50,  # 可调：返回前50个最可疑的路径
            use_softmax=True,  # True: 用 softmax 概率；False: 用 logits（注意：logits 差值可能更鲁棒）
            class_names=None  # 可选：用于日志打印（如 ['normal', 'abnormal']）
        )
        return [acc1, acc2, c_m], loss, center_loss, auc, wrong_0_to_1_paths, wrong_1_to_0_paths
    def validate(self, epoch, mode="val"):
        """Validate the model for one epoch
        """
        self.model.eval()
        all_pred, all_target, allpath, all_output = [], [], [], []
        all_loss, all_center_loss = 0, 0
        # self.ema.apply_shadow()
        if mode == "val":
            dataloader = self.val_dataloader
        elif mode == "test":
            dataloader = self.test_dataloader
        for i, (images, pathname, target) in enumerate(dataloader):
            print("Testing epoch \t{}: {}\\{}".format(
                epoch, i + 1, len(dataloader)), end='\r')
            images = images.to(self.device)
            target = target.to(self.device)
            center_loss = 0
            with torch.no_grad():
                if self.args.center_loss:
                    output, feature = self.model(images)
                    center_loss = self.model.center_loss(feature, target)
                else:
                    output = self.model(images)
            if output.dim() == 1:
                output = output.unsqueeze(0)
            loss = self.criterion(output, target)
            pred = torch.argmax(output, 1).cpu().detach().numpy()
            target = target.cpu().numpy()
            all_pred.extend(pred)
            all_target.extend(target)
            all_loss += loss.item()
            if self.args.center_loss:
                all_center_loss += center_loss.item()
            all_output.append(output.cpu())
            allpath.extend(pathname)
        all_output = torch.cat(all_output, dim=0).cpu()
        probs = F.softmax(all_output, dim=1)[:, 1].numpy()  # 正类（class 1）概率
        auc = roc_auc_score(all_target, probs)
        acc1 = accuracy_score(all_target, all_pred)         # WAR
        acc2 = balanced_accuracy_score(all_target, all_pred)         # UAR
        c_m = confusion_matrix(all_target, all_pred)
        loss = all_loss / len(dataloader)
        center_loss = all_center_loss / len(dataloader)
        wrong_0_to_1_paths, wrong_1_to_0_paths = get_wrong_path_with_confidence(
            all_pred, all_target, allpath, all_output,
            top_k=50,  # 可调：返回前50个最可疑的路径
            use_softmax=True,  # True: 用 softmax 概率；False: 用 logits（注意：logits 差值可能更鲁棒）
            class_names=None  # 可选：用于日志打印（如 ['normal', 'abnormal']）
        )
        # self.ema.restore()
        return [acc1, acc2, c_m], loss, center_loss, auc, wrong_0_to_1_paths, wrong_1_to_0_paths
    def save(self, state, is_best):
        # save the best model
        if is_best:
            checkpoint_path = os.path.join(
                self.args.output_path, "model_best.pth")
            torch.save(state, checkpoint_path)
        # save the latest model for resume
        checkpoint_path = os.path.join(
            self.args.output_path, "model_latest.pth")
        torch.save(state, checkpoint_path)
    def get_acc_msg(self, epoch, train_acc, train_loss, val_acc, val_loss, best_wa, best_ua, epoch_time):
        msg = """\nEpoch {} Train\t: WA:{:.2%}, \tUA:{:.2%}, \tloss:{:.4f}
                Epoch {} Test\t: WA:{:.2%}, \tUA:{:.2%}, \tloss:{:.4f}
                Epoch {} Best\t: WA:{:.2%}, \tUA:{:.2%}
                Epoch {} Time\t: {:.1f}s\n\n""".format(epoch, train_acc[0], train_acc[1], train_loss,
                                                        epoch, val_acc[0], val_acc[1], val_loss,
                                                        epoch, best_wa, best_ua, epoch, epoch_time)
        return msg
    def get_acc_msg_val(self, epoch, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss,
                        best_test_wa, best_test_ua, epoch_time):
        msg = """\nEpoch {} Train\t: WA:{:.2%}, \tUA:{:.2%}, \tloss:{:.4f}
                Epoch {} VaL\t: WA:{:.2%}, \tUA:{:.2%}, \tloss:{:.4f}
                Epoch {} Test\t: WA:{:.2%}, \tUA:{:.2%}, \tloss:{:.4f}
                Epoch {} Best\t: WA:{:.2%}, \tUA:{:.2%}
                Epoch {} Time\t: {:.1f}s\n\n""".format(epoch, train_acc[0], train_acc[1], train_loss,
                                                        epoch, val_acc[0], val_acc[1], val_loss,
                                                        epoch, test_acc[0], test_acc[1], test_loss,
                                                        epoch, best_test_wa, best_test_ua, epoch, epoch_time)
        return msg
    def get_confusion_msg(self, confusion_matrix, val=True,
                        wrong_0_to_1_paths=None, wrong_1_to_0_paths=None):
        # 初始化空列表，避免 None 引发错误
        wrong_0_to_1_paths = wrong_0_to_1_paths or []
        wrong_1_to_0_paths = wrong_1_to_0_paths or []
        # 构建混淆矩阵文本
        if val:
            msg = "Test Confusion Matrix:\n"
        else:
            msg = "Validation Confusion Matrix:\n"
        # 打印混淆矩阵表格（原逻辑）
        for i in range(len(confusion_matrix)):
            msg += self.emotions[i]
            for cell in confusion_matrix[i]:
                msg += "\t" + str(cell)
            msg += "\n"
        for emotion in self.emotions:
            msg += "\t" + emotion
        msg += "\n\n"
        # 计算指标（原逻辑）
        TN = confusion_matrix[0, 0]  # True Negatives (Normal → Normal)
        FP = confusion_matrix[0, 1]  # False Positives (Normal → Palsy)
        FN = confusion_matrix[1, 0]  # False Negatives (Palsy → Normal)
        TP = confusion_matrix[1, 1]  # True Positives (Palsy → Palsy)
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)  # 防除零
        sensitivity = TP / (TP + FN + 1e-8)
        specificity = TN / (TN + FP + 1e-8)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        msg += f"Accuracy: {accuracy:.4f}\n"
        msg += f"Sensitivity (Recall): {sensitivity:.4f}\n"
        msg += f"Specificity: {specificity:.4f}\n"
        # ✅ 新增：添加错误样本路径（关键修改！）
        msg += "\n" + "="*60 + "\n"
        msg += "❌ MISCLASSIFIED SAMPLES (for debugging):\n"
        msg += "="*60 + "\n"
        # 0→1: Normal → Palsy (False Positive)
        msg += f"\n🔹 [False Positives] Normal samples misclassified as Palsy ({len(wrong_0_to_1_paths)}):\n"
        if wrong_0_to_1_paths:
            # 取前 10 条 + 提示总数（防日志过长）
            for i, p in enumerate(wrong_0_to_1_paths[:10], 1):
                # 简洁显示：保留最后两级目录/文件名（可选美化）
                short_p = "/".join(p.split("/")[-2:]) if len(p.split("/")) > 2 else p
                msg += f"  {i:2d}. {short_p}\n"
            if len(wrong_0_to_1_paths) > 10:
                msg += f"  ... and {len(wrong_0_to_1_paths) - 10} more.\n"
        else:
            msg += "  (None)\n"
        # 1→0: Palsy → Normal (False Negative)
        msg += f"\n🔹 [False Negatives] Palsy samples misclassified as Normal ({len(wrong_1_to_0_paths)}):\n"
        if wrong_1_to_0_paths:
            for i, p in enumerate(wrong_1_to_0_paths[:5], 1):
                short_p = "/".join(p.split("/")[-2:]) if len(p.split("/")) > 2 else p
                msg += f"  {i:2d}. {short_p}\n"
            if len(wrong_1_to_0_paths) > 5:
                msg += f"  ... and {len(wrong_1_to_0_paths) - 5} more.\n"
        else:
            msg += "  (None)\n"
        msg += "\n" + "="*60 + "\n\n"
        return msg
    def get_confusion_msg_easy(self, confusion_matrix, val=True, auc=0,
                            wrong_0_to_1_paths=None, wrong_1_to_0_paths=None):
        # 初始化空列表，避免 None 引发错误
        wrong_0_to_1_paths = wrong_0_to_1_paths or []
        wrong_1_to_0_paths = wrong_1_to_0_paths or []

        if val:
            msg = "Test Confusion Matrix:\n"
        else:
            msg = "Validation Confusion Matrix:\n"
        # 打印混淆矩阵表格
        for i in range(len(confusion_matrix)):
            msg += self.emotions[i]
            for cell in confusion_matrix[i]:
                msg += "\t" + str(cell)
            msg += "\n"
        for emotion in self.emotions:
            msg += "\t" + emotion
        msg += "\n\n"
        # 提取二分类混淆矩阵元素（假设 0=正常, 1=面瘫）
        TN = confusion_matrix[0, 0]  # True Negative
        FP = confusion_matrix[0, 1]  # False Positive
        FN = confusion_matrix[1, 0]  # False Negative
        TP = confusion_matrix[1, 1]  # True Positive
        # 防除零
        eps = 1e-8
        accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
        sensitivity = TP / (TP + FN + eps)          # Recall
        specificity = TN / (TN + FP + eps)
        precision = TP / (TP + FP + eps)            # 新增
        # f1 = 2 * (precision * sensitivity) / (precision + sensitivity + eps)  # 新增
        # 打印并记录
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"AUC: {auc:.4f}")
        # print(f"F1-score: {f1:.4f}")
        msg += f"Accuracy: {accuracy:.4f}\n"
        msg += f"Sensitivity (Recall): {sensitivity:.4f}\n"
        msg += f"Specificity: {specificity:.4f}\n"
        msg += f"Precision: {precision:.4f}\n"
        msg += f"AUC: {auc:.4f}\n"
        # msg += f"F1-score: {f1:.4f}\n"
        return msg


