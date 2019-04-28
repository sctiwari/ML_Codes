from __future__ import print_function
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import os
from torch.autograd import Variable
from torchvision.models.vgg import vgg16
from srgan_model import Generator, Discriminator


class SRGANTrainer(object):
    def __init__(self, config, training_loader, testing_loader, class_name):
        super(SRGANTrainer, self).__init__()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU_IN_USE else 'cpu')
        self.net_G = None
        self.net_D = None
        self.lr = config.lr
        self.num_epoch = config.num_epoch
        self.epoch_pretrain = 10
        self.loss_G = None
        self.loss_D = None
        self.optimizer_G = None
        self.optimizer_D = None
        self.feature_extractor = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.num_residuals = 16
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.g_model_out_path = "SRGAN_Generator_model_" + class_name
        self.d_model_out_path = "SRGAN_Discriminator_model_" + class_name
        self.loss_set = []
        self.psnr_set = []
        self.mse_set = []
        self.class_name = class_name
        self.num_input = 1 if self.class_name != 'velocity' else 3

    def build_model(self):
        self.net_G = Generator(num_residual=self.num_residuals,
                               upscale_factor=self.upscale_factor,
                               base_filter=128,
                               num_input=self.num_input).to(self.device)
        self.net_D = Discriminator(base_filter=128, num_input=self.num_input).to(self.device)
        #self.feature_extractor = vgg16(pretrained=True)
        self.net_G.weight_init(mean=0.0, std=0.2)
        self.net_D.weight_init(mean=0.0, std=0.2)
        self.loss_G = nn.MSELoss()
        self.loss_D = nn.BCELoss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            #self.feature_extractor.cuda()
            cudnn.benchmark = True
            self.loss_G.cuda()
            self.loss_D.cuda()

        self.optimizer_G = optim.Adam(self.net_G.parameters(),
                                      lr=self.lr,
                                      betas=(0.9, 0.999),
                                      weight_decay=1e-8)

        self.optimizer_D = optim.SGD(self.net_D.parameters(),
                                     lr=self.lr / 100,
                                     momentum=0.9,
                                     nesterov=True)

        '''
        self.optimizer_D = optim.Adam(self.net_D.parameters(),
                                     lr=self.lr / 100,
                                      betas=(0.9, 0.999),
                                      weight_decay=1e-8)
        '''
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer_G,
                                                        milestones=[20, 40, 60, 80, 100],
                                                        gamma=0.5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer_D,
                                                        milestones=[20, 40, 60, 80, 100],
                                                        gamma=0.5)

    @staticmethod
    def to_data(x):
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def save(self):
        torch.save(self.net_G, self.g_model_out_path)
        torch.save(self.net_D, self.d_model_out_path)
        print("Checkpoint saved to {}".format(self.g_model_out_path))
        print("Checkpoint saved to {}".format(self.d_model_out_path))

    def save_loss(self, cnt):
        np.save("loss_set_" + self.class_name + str(cnt), np.array(self.loss_set))
        np.save("psnr_set_" + self.class_name + str(cnt), np.array(self.psnr_set))
        np.save("mse_set_" + self.class_name + str(cnt), np.array(self.mse_set))

    def load(self):
        self.net_G = torch.load(self.g_model_out_path)
        self.net_D = torch.load(self.d_model_out_path)

    def pretrain(self):
        self.net_G.train()
        for batch_num, (data, target) in enumerate(self.training_loader):
            print("batch_num: ", batch_num, "/", len(self.training_loader) - 1)
            data, target = data.to(self.device), target.to(self.device)
            #data, target = Variable(data).cuda(), Variable(target).cuda()
            #print(data[0][0].shape)
            #print(data[1][0].shape)
            #print(data[2][0].shape)
            #print(data[3][0].shape)
            self.net_G.zero_grad()
            gen = self.net_G(data)
            loss = self.loss_G(gen, target.float())
            loss.backward()
            self.optimizer_G.step()
        torch.cuda.empty_cache()

    def train(self):
        # models setup
        self.net_G.train()
        self.net_D.train()
        g_train_loss = 0
        d_train_loss = 0
        torch.cuda.empty_cache()
        for batch_num, (data, target) in enumerate(self.training_loader):
            real_label = torch.ones(data.size(0), data.size(1)).to(self.device)
            fake_label = torch.zeros(data.size(0), data.size(1)).to(self.device)
            data, target = data.to(self.device), target.to(self.device)

            # Discriminator
            self.optimizer_D.zero_grad()
            d_real = self.net_D(target.float())
            d_real_loss = self.loss_D(d_real, real_label)

            d_fake = self.net_D(self.net_G(data))
            d_fake_loss = self.loss_D(d_fake, fake_label)
            print(d_real_loss, d_fake_loss)
            d_total = d_real_loss + d_fake_loss
            d_train_loss += d_total.item()
            d_total.backward()
            self.optimizer_D.step()

            # Generator
            self.optimizer_G.zero_grad()
            g_real = self.net_G(data)
            print(g_real.shape)
            print(target.shape)
            g_fake = self.net_D(g_real)
            gan_loss = self.loss_D(g_fake, real_label)
            mse_loss = self.loss_G(g_real, target.float())

            print (mse_loss, gan_loss)
            g_total = mse_loss + 0.001 * gan_loss
            #g_total = gan_loss
            g_train_loss += g_total.item()
            g_total.backward()
            self.optimizer_G.step()

            #progress_bar(batch_num, len(self.training_loader), 'G_Loss: %.4f | D_Loss: %.4f' % (g_train_loss / (batch_num + 1), d_train_loss / (batch_num + 1)))
        average_loss = g_train_loss / len(self.training_loader)
        print("    Average G_Loss: {:.8f}".format(average_loss))
        return average_loss

    def test(self):
        self.net_G.eval()
        avg_psnr = 0
        avg_mse_loss = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)

                prediction = self.net_G(data)
                mse = self.loss_G(prediction, target.float())

                psnr = 10 * log10(1 / mse.item())

                avg_mse_loss += mse.item()
                avg_psnr += psnr
                #progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))
        average_psnr = avg_psnr / len(self.testing_loader)
        average_mse = avg_mse_loss / len(self.testing_loader)
        print("    Average MSE Loss: {:.8f}".format(average_mse))
        print("    Average PSNR: {:.8f} dB".format(average_psnr))
        return average_psnr, average_mse

    def run(self):
        self.build_model()
        '''
        if (self.class_name != 'velocity'):
            for epoch in range(1, self.epoch_pretrain + 1):
                self.pretrain()
                print("{}/{} pretrained".format(epoch, self.epoch_pretrain))
        '''
        for epoch in range(1, self.num_epoch + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            loss = self.train()
            psnr, mse_loss = self.test()
            self.scheduler.step(epoch)
            self.loss_set.append(loss)
            self.psnr_set.append(psnr)
            self.mse_set.append(mse_loss)
            if epoch % 10 == 0:
                self.save_loss(epoch)
                self.save()
            if epoch == self.num_epoch:
                self.save()

    def restore(self):
        self.build_model()
        self.load()
