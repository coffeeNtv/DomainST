import os
import torch
import numpy as np
import argparse


def list_sorted_subfolders(folder_path):
    # obtain all sub folders
    subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
    # sort by name
    sorted_subfolders = sorted(subfolders)
    return [os.path.join(folder_path, i) for i in sorted_subfolders]


# DomainST result computation
def patient_results(path):
    mae = torch.load(os.path.join(path, 'MAE'))
    mse = torch.load(os.path.join(path, 'MSE'))
    cor = torch.load(os.path.join(path, 'cor'))
    return mae, mse, np.nanmean(cor, axis=0)


def metrics(dataset, output_dir):

    res_subfolders = list_sorted_subfolders(output_dir)
    pcc = []
    mse = []
    mae = []
    for i in range(len(res_subfolders)):
        mae_tmp, mse_tmp, pcc_tmp = patient_results(res_subfolders[i])
        pcc.append(pcc_tmp)
        mse.append(mse_tmp)
        mae.append(mae_tmp)

    # find top 50 genes
    pcc_50 = []
    for i in range(len(res_subfolders)):
        pcc_50.append(torch.load(os.path.join(res_subfolders[i], 'cor')))

    # init tensor for storing rankings
    rank_sum = torch.zeros(250)

    # compute rank and sum
    for tensor in pcc_50:
        ranks = torch.argsort(tensor, dim=0)  # get rank, start from 0, ascending order
        for i in range(len(ranks)):
            rank_sum[ranks[i]] += i  # for index has lower value(ranks[i]), its rank i is lower as well

    top_50_indices = torch.argsort(rank_sum, dim=0, descending=True)[:50]

    # compute pcc for top 50 gene 
    top_50_genes = [all_genes[top_50_indices].numpy() for all_genes in pcc_50]

    patient_pcc = [np.nanmean(tmp, axis=0) for tmp in top_50_genes]
    result_path = os.path.join(output_dir,f"{output_dir.split('/')[-2]}.csv")

    mae_m = np.nanmean(mae)
    mse_m = np.nanmean(mse)
    p_250_m = np.nanmean(pcc)
    p_50_m = np.nanmean(patient_pcc)

    mae_s = np.std(mae)
    mse_s = np.std(mse)
    p_250_s = np.std(pcc)
    p_50_s = np.std(patient_pcc)

    import csv
    with open(result_path, "w", newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["mae_m", "mae_s", "mse_m", "mse_s", "p_250_m", "p_250_s", "p_50_m", "p_50_s"])
        writer.writerow([f"{mae_m:.3f}", f"{mae_s:.3f}", f"{mse_m:.3f}",
                         f"{mse_s:.3f}", f"{p_250_m:.3f}", f"{p_250_s:.3f}",
                         f"{p_50_m:.3f}", f"{p_50_s:.3f}"])

    print(f'dataset: {dataset}')
    print("MAE mean: %.3f" % mae_m)
    print("MSE mean: %.3f" % mse_m)
    print("PCC(250) mean: %.3f" % p_250_m)
    print("PCC(50) mean: %.3f" % p_50_m)



def eval_model(dataset, encoder='res50', gpu=0, model_name="test"):

    FOLD_DICT = {'her2st': 8, 'skin': 4, 'stnet': 8}
    output_dir = os.path.join('results', model_name)

    if os.path.exists(output_dir):
        for fold in range(FOLD_DICT[dataset]):
            cur_dir = os.path.join('logs', model_name, f'{dataset}_fold_{fold}')
            for f in os.listdir(cur_dir):
                if f.endswith('.ckpt'):
                    cmd = f"python main.py --config {dataset}/DomainST --mode test --fold {fold} --model_path " + \
                          f"{os.path.join(cur_dir, f)} --encoder {encoder}  --gpu {gpu} --model_name {model_name}"
                    print(cmd)
                    os.system(cmd)
    else:
        print("output dir not found!")

if __name__ == '__main__':

    # Parsing the command-line arguments
    parser = argparse.ArgumentParser(description='evaluate model by given parameters')
    parser.add_argument('--dataset', type=str, required=True, choices=['her2st', 'skin', 'stnet'], help='datasets')
    parser.add_argument('--encoder', type=str, default='res50', help='encoder')
    parser.add_argument('--gpu', type=int, default=7, help='gpu id')
    parser.add_argument('--model_name', type=str, help='model name')
    args = parser.parse_args()

    eval_model(dataset=args.dataset, encoder=args.encoder, gpu=args.gpu, model_name=args.model_name)
    
    metrics(args.dataset, os.path.join('results', args.model_name, args.dataset))
