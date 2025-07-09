import torch
import numpy as np
import argparse
import time
import os
import util
from engine import trainer
from model import PathChoiceDecoder, SequentialConsistencyLoss

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0', help='which gpu')
parser.add_argument('--data', type=str, default='data/SiouxFalls', help='data path')
parser.add_argument('--adjdata', type=str, default='data/SiouxFalls/adj_mx.pkl', help='adj data path')
parser.add_argument('--seq_length', type=int, default=12, help='forecast horizon')
parser.add_argument('--nhid', type=int, default=40)
parser.add_argument('--in_dim', type=int, default=1)
parser.add_argument('--num_nodes', type=int, default=24)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--clip', type=int, default=3)
parser.add_argument('--lr_decay_rate', type=float, default=0.97)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--top_k', type=int, default=4)
parser.add_argument('--print_every', type=int, default=50)
parser.add_argument('--save', type=str, default='./garage/metr-la')
parser.add_argument('--seed', type=int, default=530302)
args = parser.parse_args()

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    setup_seed(args.seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Data Loading
    adj_mx = util.load_adj(args.adjdata)
    supports = [torch.tensor(i).cuda() for i in adj_mx]
    H_a, H_b, H_T_new, lwjl, G0, G1, indices, G0_all, G1_all = util.load_hadj(args.adjdata, args.top_k)

    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    # Adjust the shape of lwjl
    lwjl = (((lwjl.t()).unsqueeze(0)).unsqueeze(3)).repeat(args.batch_size, 1, 1, 1)

    H_a = H_a.cuda()
    H_b = H_b.cuda()
    G0 = torch.tensor(G0).cuda()
    G1 = torch.tensor(G1).cuda()
    H_T_new = torch.tensor(H_T_new).cuda()
    lwjl = lwjl.cuda()
    indices = indices.cuda()
    G0_all = torch.tensor(G0_all).cuda()
    G1_all = torch.tensor(G1_all).cuda()

    # ===== Initialize the generator =====
    print("Creating engine(FusionTrainer)...")
    engine = trainer(
        args.batch_size, scaler, args.in_dim, args.seq_length, args.num_nodes,
        args.nhid, args.dropout, args.learning_rate, args.weight_decay,
        supports, H_a, H_b, G0, G1, indices, G0_all, G1_all, H_T_new, lwjl,
        args.clip, args.lr_decay_rate
    )

    # ===== Create PathChoiceDecoder and SequentialConsistencyLoss =====
    path_choice_decoder = PathChoiceDecoder(input_size=args.nhid, hidden_size=64, output_size=args.seq_length).cuda()
    consistency_loss_fn = SequentialConsistencyLoss(lambda_consistency=0.1).cuda()

    print("start training...")
    his_loss = []
    val_time = []
    train_time = []

    for i in range(1, args.epochs + 1):
        t1 = time.time()
        train_loss = []
        train_mape = []
        train_rmse = []

        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).cuda().transpose(1, 3)     # => (B,1,N,T)
            trainy = torch.Tensor(y).cuda().transpose(1, 3)     # => (B,1,N,T)

            metrics = engine.train_step(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[3])
            train_rmse.append(metrics[4])

            if iter % args.print_every == 0:
                print(f"Iter: {iter}, Train Loss: {metrics[0]:.4f}, "
                      f"D Loss: {metrics[1]:.4f}, DRF Loss: {metrics[2]:.4f}, "
                      f"MAPE: {metrics[3]:.4f}, RMSE: {metrics[4]:.4f}")

        t2 = time.time()
        train_time.append(t2 - t1)

        if engine.scheduler is not None:
            engine.scheduler.step()

        # 验证集
        s1 = time.time()
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).cuda().transpose(1, 3)   # => (B,1,N,T)
            testy = torch.Tensor(y).cuda().transpose(1, 3)   # => (B,1,N,T)
            metrics = engine.eval_step(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])

        s2 = time.time()
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)

        his_loss.append(mvalid_loss)

        print(f"Epoch: {i}, Train Loss: {mtrain_loss:.4f}, Valid Loss: {mvalid_loss:.4f}, "
              f"Train MAPE: {mtrain_mape:.4f}, Valid MAPE: {mvalid_mape:.4f}, "
              f"Train RMSE: {mtrain_rmse:.4f}, Valid RMSE: {mvalid_rmse:.4f}, "
              f"Training Time: {t2 - t1:.2f}s")

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    bestid = np.argmin(his_loss)
    print("Best model epoch:", bestid+1, "  val_loss=", his_loss[bestid])

    # Load the best weights
    # engine.generator.load_state_dict(torch.load(...))

    # Inference on the test set
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).cuda().transpose(1, 3)[:, 0, :, :]  # (B,N,T)
    with torch.no_grad():
        for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).cuda().transpose(1, 3)
            preds = engine.generator(testx).transpose(1, 3)  # => (B,1,N,T)
            outputs.append(preds.squeeze(1))                 # => (B,N,T)

    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]  # => (B,N,T)

    # Compute the error at each time step step by step
    amae, amape, armse = [], [], []
    for i in range(args.seq_length):
        pred = dataloader['scaler'].inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        mae, mape, rmse = util.metric(pred, real)
        amae.append(mae)
        amape.append(mape)
        armse.append(rmse)
        print(f"Horizon {i+1}: MAE {mae:.4f}, MAPE {mape:.4f}, RMSE {rmse:.4f}")

    print("Average: MAE {:.4f}, MAPE {:.4f}, RMSE {:.4f}".format(
        np.mean(amae), np.mean(amape), np.mean(armse)))

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"Total training time: {t2 - t1:.2f} seconds.")
