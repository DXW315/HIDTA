import torch
import torch.nn as nn
import torch.optim as optim
from model import HIDTA
from util import masked_mae, masked_mape, masked_rmse, StandardScaler

def build_discriminator(input_dim):
    """
Dynamically create the discriminator to adapt to the (B*N, input_dim) input
    """
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

class FusionTrainer:
    def __init__(self, batch_size, scaler, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay,
                 supports, H_a, H_b, G0, G1, indices, G0_all, G1_all, H_T_new, lwjl,
                 lambda_gan1=0.01, lambda_gan2=0.01, clip=3, lr_de_rate=0.97):

        # ===== Initialize the generator =====
        self.generator = HIDTA(
            batch_size, H_a, H_b, G0, G1, indices, G0_all, G1_all, H_T_new, lwjl,
            num_nodes, dropout, supports=supports, in_dim=in_dim, out_dim=seq_length,
            residual_channels=nhid, dilation_channels=nhid,
            skip_channels=nhid * 8, end_channels=nhid * 16
        ).cuda()

        # ===== Internal variables of the trainer =====
        self.discriminator = None
        self.discriminator_rf = None
        self.scaler = scaler

        self.loss_fn = masked_mae
        self.loss_bce = nn.BCELoss()

        self.clip = clip
        self.lambda_gan1 = lambda_gan1
        self.lambda_gan2 = lambda_gan2
        self.lrate = lrate
        self.wdecay = wdecay
        self.lr_de_rate = lr_de_rate

        self.optim_G = None
        self.optim_D = None
        self.optim_DRF = None
        self.scheduler = None

    def train_step(self, x, y):

        self.generator.train()

        # x: (B,1,N,T+1)
        # y: (B,N,T)


        x = nn.functional.pad(x, (1, 0, 0, 0))  # => (B,1,N,T+2)
        real = torch.unsqueeze(y, dim=1)        # => (B,1,N,T)

        # Forward generation
        fake = self.generator(x).transpose(1, 3)   # => (B,1,N,T)
        pred = self.scaler.inverse_transform(fake) # => (B,1,N,T)
        real_score = self.scaler.inverse_transform(real)

        B, _, N, T = pred.shape

        # ===== Discriminator input 1: concat => (B,1,N,2T+1) => (B,N,2T+1) => reshape => (B*N,2T+1)
        cat_pred = torch.cat([x, pred], dim=3)   # => (B,1,N,(T+2)+T = 2T+2)
        cat_pred = cat_pred.squeeze(1)           # => (B,N,2T+2)
        cat_real = torch.cat([x, real_score], dim=3).squeeze(1)

        disc_in_pred = cat_pred.reshape(B*N, -1)  # => (B*N,2T+1或2T+2)
        disc_in_real = cat_real.reshape(B*N, -1)

        # =====  Discriminator input  2: flow => (B, N, T) => (B*N,T)
        pred_flow = pred.squeeze(1)        # => (B,N,T)
        real_flow = real_score.squeeze(1)  # => (B,N,T)

        disc_in_flow_fake = pred_flow.reshape(B*N, -1)
        disc_in_flow_real = real_flow.reshape(B*N, -1)

        # If the discriminator is not initialized, dynamically construct it here
        if self.discriminator is None:
            disc_dim_1 = disc_in_pred.shape[-1]   # 2T+1 or 2T+2
            disc_dim_2 = disc_in_flow_fake.shape[-1]  # T
            self.discriminator = build_discriminator(disc_dim_1).cuda()
            self.discriminator_rf = build_discriminator(disc_dim_2).cuda()

            self.optim_D = optim.Adam(self.discriminator.parameters(), lr=self.lrate * 0.1)
            self.optim_DRF = optim.Adam(self.discriminator_rf.parameters(), lr=self.lrate * 0.1)
            self.optim_G = optim.Adam(self.generator.parameters(), lr=self.lrate, weight_decay=self.wdecay)
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optim_G, lr_lambda=lambda epoch: self.lr_de_rate ** epoch
            )

        # Prepare the labels
        valid = torch.ones(B*N, 1).cuda()
        fake_label = torch.zeros(B*N, 1).cuda()

        # ===== Generator optimization =====
        self.optim_G.zero_grad()

        loss_pred = self.loss_fn(pred, real, 0.0)
        loss_gan1 = self.loss_bce(self.discriminator(disc_in_pred), valid)
        loss_gan2 = self.loss_bce(self.discriminator_rf(disc_in_flow_fake), valid)

        loss_G = loss_pred + self.lambda_gan1 * loss_gan1 + self.lambda_gan2 * loss_gan2
        loss_G.backward()
        if self.clip:
            nn.utils.clip_grad_norm_(self.generator.parameters(), self.clip)
        self.optim_G.step()

        # ===== Discriminator 1 optimization =====
        self.optim_D.zero_grad()
        loss_real = self.loss_bce(self.discriminator(disc_in_real), valid)
        loss_fake = self.loss_bce(self.discriminator(disc_in_pred.detach()), fake_label)
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        self.optim_D.step()

        # ===== Discriminator 2 optimization =====
        self.optim_DRF.zero_grad()
        loss_rf_real = self.loss_bce(self.discriminator_rf(disc_in_flow_real), valid)
        loss_rf_fake = self.loss_bce(self.discriminator_rf(disc_in_flow_fake.detach()), fake_label)
        loss_DRF = 0.5 * (loss_rf_real + loss_rf_fake)
        loss_DRF.backward()
        self.optim_DRF.step()


        mape = masked_mape(pred, real, 0.0).item()
        rmse = masked_rmse(pred, real, 0.0).item()

        return loss_G.item(), loss_D.item(), loss_DRF.item(), mape, rmse

    def eval_step(self, x, y):
        self.generator.eval()
        x = nn.functional.pad(x, (1, 0, 0, 0))
        real = torch.unsqueeze(y, dim=1)

        with torch.no_grad():
            fake = self.generator(x).transpose(1, 3)
            pred = self.scaler.inverse_transform(fake)
            real_score = self.scaler.inverse_transform(real)

        loss = self.loss_fn(pred, real_score, 0.0)
        mape = masked_mape(pred, real_score, 0.0).item()
        rmse = masked_rmse(pred, real_score, 0.0).item()
        return loss.item(), mape, rmse

trainer = FusionTrainer
