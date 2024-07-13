from __future__ import print_function
import argparse
from math import log10, ceil
import random, shutil, json
from collections import OrderedDict
from os.path import join, exists, isfile, realpath, dirname, expanduser
from os import makedirs, remove, chdir, environ

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
import h5py
import faiss

from tensorboardX import SummaryWriter
import numpy as np
import netvlad

parser = argparse.ArgumentParser(description='pytorch-NetVLAD')

parser.add_argument('--mode', type=str, default='cluster', help='Mode', choices=['train', 'test', 'cluster'])
parser.add_argument('--batch_size', type=int, default=4, 
        help='Number of triplets (query, pos, negs). Each triplet consists of 12 images.')
parser.add_argument('--cache_batch_size', type=int, default=24, help='Batch size for caching and testing')
parser.add_argument('--cache_refresh_rate', type=int, default=1000, 
        help='How often to refresh cache, in number of queries. 0 for off')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', 
        help='manual epoch number (useful on restarts)')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPU to use.')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use', choices=['SGD', 'ADAM'])
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
parser.add_argument('--lrStep', type=float, default=5, help='Decay LR ever N steps.')
parser.add_argument('--lrGamma', type=float, default=0.5, help='Multiply LR by Gamma for decaying.')
parser.add_argument('--weightDecay', type=float, default=0.001, help='Weight decay for SGD.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--threads', type=int, default=8, help='Number of threads for each data loader to use')
parser.add_argument('--seed', type=int, default=123, help='Random seed to use.')
parser.add_argument('--dataPath', type=str, default=join(expanduser("~"), "pytorch-NetVlad", "data"), help='Path for centroid data.')
parser.add_argument('--runsPath', type=str, default=join(expanduser("~"), "pytorch-NetVlad", "data", "runs"), help='Path to save runs to.')
parser.add_argument('--save_path', type=str, default='checkpoints', 
        help='Path to save checkpoints to in logdir. Default=checkpoints/')
parser.add_argument('--cache_path', type=str, default=join(expanduser("~"), "pytorch-NetVlad", "data", "temp"), help='Path to save cache to.')
parser.add_argument('--resume', type=str, default='', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--ckpt', type=str, default='latest', 
        help='Resume from latest or best checkpoint, or given path.', choices=['latest', 'best', 'path'])
parser.add_argument('--evalEvery', type=int, default=1, 
        help='Do a validation set run, and save, every N epochs.')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping. 0 is off.')
parser.add_argument('--dataset', type=str, default='fourseasons', 
        help='Dataset to use', choices=['fourseasons'])
parser.add_argument('--arch', type=str, default='resnet50', 
        help='basenetwork to use', choices=['vgg16', 'alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
        choices=['netvlad', 'max', 'avg'])
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss. Default=0.1')
parser.add_argument('--val', type=str, default='train', help='Data split to use for testing. Default is val', 
        choices=['test', 'train', 'val'])
parser.add_argument('--fromscratch', action='store_true', help='Train from scratch rather than using pretrained models')
parser.add_argument('--storetest', action='store_true', help='Store infered descriptors if mode=test')
parser.add_argument('--ckptpath', type=str, default=None, help='Path for loading model ckpt if ckpt=path')

def train(epoch):

    epoch_loss = 0
    iteration_index = 1 

    if opt.cache_refresh_rate > 0:
        # number of queries in total divided by rate how often to refresh
        # i.e. how many subsets will there be in total?
        subset_n = ceil(len(train_set) / opt.cache_refresh_rate)
        # split list of query indices into subset_n many lists (of almost equal size)
        # TODO: randomise the arange before splitting?
        subset_idx = np.array_split(np.arange(len(train_set)), subset_n)
    else:
        # if cache_refresh_rate = 0 then only one subset
        subset_n = 1
        subset_idx = [np.arange(len(train_set))]

    # how many batches in total (batch defined by multiple query triples)
    num_batches = (len(train_set) + opt.batch_size - 1) // opt.batch_size

    for subset_iteration in range(subset_n):

        print("====> Building cache")

        model.eval()
        train_set.cache = join(opt.cache_path, train_set.which_dataset + "_feat_cache.hdf5")

        with h5py.File(train_set.cache, mode="w") as h5:

            pool_size = encoder_dim

            if opt.pooling.lower() == "netvlad": pool_size *= opt.num_clusters

            h5feat = h5.create_dataset("features", 
                [len(whole_train_set), pool_size],
                dtype=np.float32)
            
            with torch.no_grad():

                for iteration, (input, indices) in tqdm(enumerate(whole_training_data_loader, 1),
                    total=ceil(len(whole_train_set)/opt.cache_batch_size)):

                    # generate encodings for each query
                    # based on "current" version of network
                    # within epoch, after cache_refresh_rate many queries the cache gets refreshed

                    input = input.to(device)
                    image_encoding = model.encoder(input)
                    vlad_encoding = model.pool(image_encoding) 
                    h5feat[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()

                    del input, image_encoding, vlad_encoding

        # load queries that are within current subset
        sub_train_set = Subset(dataset=train_set, indices=subset_idx[subset_iteration])

        training_data_loader = DataLoader(dataset=sub_train_set, num_workers=opt.threads, 
            batch_size=opt.batch_size, shuffle=True, 
            collate_fn=dataset.collate_fn, pin_memory=cuda)

        print("Allocated:", torch.cuda.memory_allocated())
        print("Cached:", torch.cuda.memory_cached())

        model.train()

        for iteration, (query, positives, negatives, neg_counts,
            indices) in tqdm(enumerate(training_data_loader, iteration_index),
            total=ceil(len(subset_idx[subset_iteration]/opt.batch_size))):
            
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (num_query + num_pos + num_neg)

            if query is None: continue # in case we get an empty batch

            B, _C, _H, _W = query.shape
            num_neg = torch.sum(neg_counts)
            input = torch.cat([query, positives, negatives])

            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding) 

            # there are B many queries with each one positive (i.e. in total B many)
            # but there are more negatives (multiple possible for one query)
            vlad_queries, vlad_positives, vlad_negatives = torch.split(vlad_encoding, [B, B, num_neg])

            optimizer.zero_grad()
            
            # calculate loss for each query, positive, negative triplet
            # due to potential difference in number of negatives have to 
            # do it per query, per negative
            loss = 0

            # iterate over list of number of negative
            for i, neg_cnt in enumerate(neg_counts):
                
                # for each query iterate over negatives
                for n in range(neg_cnt):

                    # get index in vlad_negatives
                    neg_idx = (torch.sum(neg_counts[:i]) + n).item()
                    # calculate loss for triplet (Q,P,N)
                    loss += criterion(vlad_queries[i:i+1], vlad_positives[i:i+1], vlad_negatives[neg_idx:neg_idx+1])

            # normalise by actual number of negatives
            loss /= num_neg.float().to(device)
            loss.backward()

            optimizer.step()

            del input, image_encoding, vlad_encoding, vlad_queries, vlad_positives, vlad_negatives
            del query, positives, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 50 == 0 or num_batches <= 10:

                print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, 
                    num_batches, batch_loss), flush=True)
                
                writer.add_scalar('Train/Loss', batch_loss, 
                        ((epoch-1) * num_batches) + iteration)
                writer.add_scalar('Train/num_neg', num_neg, 
                        ((epoch-1) * num_batches) + iteration)
                
                print("Allocated:", torch.cuda.memory_allocated())
                print("Cached:", torch.cuda.memory_cached())

        iteration_index += len(training_data_loader)

        del training_data_loader, loss

        optimizer.zero_grad()
        torch.cuda.empty_cache()

        # delete HDF5 cache
        remove(train_set.cache)

    avg_loss = epoch_loss / num_batches

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss), 
        flush=True)
    
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

def test(eval_set, epoch=0, write_tboard=False):
    
    # TODO: what if features dont fit in memory? 
    test_data_loader = DataLoader(dataset=eval_set, 
        num_workers=opt.threads, batch_size=opt.cache_batch_size, shuffle=False, 
        pin_memory=cuda)

    model.eval()

    with torch.no_grad():
        
        print("====> Extracting Features")

        pool_size = encoder_dim

        if opt.pooling.lower() == "netvlad": pool_size *= opt.num_clusters

        database_features = np.empty((len(eval_set), pool_size))

        for iteration, (input, indices) in enumerate(test_data_loader, 1):

            # 'infer' test queries and database image
            # i.e. for each image VLAD encoding is generated

            input = input.to(device)
            image_encoding = model.encoder(input)
            vlad_encoding = model.pool(image_encoding) 

            database_features[indices.detach().numpy(), :] = vlad_encoding.detach().cpu().numpy()

            if iteration % 50 == 0 or len(test_data_loader) <= 10:

                print("==> Batch ({}/{})".format(iteration, 
                    len(test_data_loader)), flush=True)

            del input, image_encoding, vlad_encoding

    del test_data_loader

    # extracted for both db and query, now split in own sets
    query_features = database_features[eval_set.num_db:].astype("float32")
    database_features = database_features[:eval_set.num_db].astype("float32")

    if opt.storetest: # store infered descriptors

        print("eval_set height levels db", eval_set.range_per_height_db)
        print("eval_set height levels query", eval_set.range_per_height_query)

        np.save(f"query_features_{opt.split}.npy", query_features)
        np.save(f"database_features_{opt.split}.npy", database_features)

    n_values = [1, 5, 10, 20]
    all_predictions = [] # for all height levels

    for hl in eval_set.height_levels:

        # db: |---db300---|---db400---|---db500---|---db600---|
        # '-> get offset and end for each height level
        offset_db = eval_set.range_per_height_db[hl]["start"]
        end_db = eval_set.range_per_height_db[hl]["end"]
    
        print(f"====> Building faiss index for height level {hl}")
        faiss_index = faiss.IndexFlatL2(pool_size)
        faiss_index.add(database_features[offset_db:end_db])

        # query: |---queries300---|---queries400---|---queries500---|---queries600---|
        # '-> get offset and end for each height level
        offset_query = eval_set.range_per_height_query[hl]["start"]
        end_query = eval_set.range_per_height_query[hl]["end"]

        _, predictions = faiss_index.search(query_features[offset_query:end_query],
            max(n_values))

        # predictions are now for local indices of e.g. |---queries500---|
        # '-> add offset for this database within whole db to transform to global
        # height levels are viewed in order, so query predictions can simply be added to list
        all_predictions.append(predictions + offset_db)

    # adapt to following code, which uses "predictions"
    predictions = np.vstack(all_predictions)

    print("====> Calculating recall @ N")

    # for each query get those within threshold distance
    gt = eval_set.getPositives()

    correct_at_n = np.zeros(len(n_values))

    # TODO: can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):

        for i, n in enumerate(n_values):

            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break

    recall_at_n = correct_at_n / eval_set.num_queries
    recalls = {} # make dict for output

    for i, n in enumerate(n_values):

        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))

        if write_tboard: writer.add_scalar("Val/Recall@" + str(n), recall_at_n[i], epoch)

    return recalls

def get_clusters(cluster_set, num_descriptors=50000, num_per_image=100, niter_clustering=100):

    num_images = ceil(num_descriptors / num_per_image)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), num_images, replace=False))

    data_loader = DataLoader(dataset=cluster_set, 
        num_workers=opt.threads, batch_size=opt.cache_batch_size, shuffle=False, 
        pin_memory=cuda,
        sampler=sampler)

    if not exists(join(opt.dataPath, "centroids")):
        makedirs(join(opt.dataPath, "centroids"))

    initcache = join(opt.dataPath, "centroids", opt.arch + "_" + cluster_set.dataset + "_" + str(opt.num_clusters) + "_desc_cen.hdf5")
    
    with h5py.File(initcache, mode="w") as h5: 

        with torch.no_grad():

            model.eval()

            print("====> Extracting Descriptors")

            database_features = h5.create_dataset("descriptors", 
                [num_descriptors, encoder_dim], 
                dtype=np.float32)

            for iteration, (input, _indices) in enumerate(data_loader, 1):

                input = input.to(device)
                image_descriptors = model.encoder(input).view(input.size(0), encoder_dim, -1).permute(0, 2, 1)

                batchix = (iteration - 1) * opt.cache_batch_size * num_per_image

                for ix in range(image_descriptors.size(0)):

                    # sample different location for each image in batch
                    sample = np.random.choice(image_descriptors.size(1), num_per_image, replace=False)
                    startix = batchix + ix*num_per_image
                    database_features[startix:startix+num_per_image, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()

                if iteration % 50 == 0 or len(data_loader) <= 10:

                    print("==> Batch ({}/{})".format(iteration, 
                        ceil(num_images / opt.cache_batch_size)), flush=True)

                del input, image_descriptors
        
        print("====> Clustering..")
        kmeans = faiss.Kmeans(encoder_dim, opt.num_clusters, niter=niter_clustering, verbose=False)
        kmeans.train(database_features[...])

        print("====> Storing centroids", kmeans.centroids.shape)

        h5.create_dataset("centroids", data=kmeans.centroids)

        print("====> Done!")

def save_checkpoint(state, is_best):

    filename = f"checkpoint_{state['epoch']}.pth.tar"

    model_out_path = join(opt.save_path, filename)

    torch.save(state, model_out_path)

    if is_best: shutil.copyfile(model_out_path, join(opt.save_path, "model_best.pth.tar"))

class Flatten(nn.Module):

    def forward(self, input):

        return input.view(input.size(0), -1)

class L2Norm(nn.Module):

    def __init__(self, dim=1):

        super().__init__()
        self.dim = dim

    def forward(self, input):

        return F.normalize(input, p=2, dim=self.dim)

if __name__ == "__main__":

    opt = parser.parse_args()

    restore_var = ["lr", "lrStep", "lrGamma", "weightDecay", "momentum", 
        "runsPath", "save_path", "arch", "num_clusters", "pooling", "optim",
        "margin", "seed", "patience"]

    if opt.resume:

        flag_file = join(opt.resume, "checkpoints", "flags.json")

        if exists(flag_file):

            with open(flag_file, "r") as f:

                stored_flags = {"--" + k : str(v) for k,v in json.load(f).items() if k in restore_var}
                
                to_del = []

                for flag, val in stored_flags.items():

                    for act in parser._actions:

                        if act.dest == flag[2:]:

                            # store_true / store_false args don't accept arguments, filter these 
                            if type(act.const) == type(True):

                                if val == str(act.default): to_del.append(flag)
                                else: stored_flags[flag] = ""

                for flag in to_del: del stored_flags[flag]

                train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
                print("Restored flags:", train_flags)
                opt = parser.parse_args(train_flags, namespace=opt)

    print(opt)

    if opt.dataset.lower() == "fourseasons": import fourseasons as dataset
    else: raise Exception("Unknown dataset")

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print("===> Loading dataset(s)")

    if opt.mode.lower() == "train":

        whole_train_set = dataset.get_whole_training_set()
        whole_training_data_loader = DataLoader(dataset=whole_train_set, 
            num_workers=opt.threads, batch_size=opt.cache_batch_size, shuffle=False, 
            pin_memory=cuda)

        train_set = dataset.get_training_query_set(opt.margin)
        whole_test_set = dataset.get_whole_val_set(which_dataset="train")

        print("====> Training query set:", len(train_set))
        print(f"===> Evaluating on {opt.split} set, query count:", whole_test_set.num_queries)

    elif opt.mode.lower() == "test":

        if opt.split.lower() == "test":
            whole_test_set = dataset.get_whole_test_set()
            print("===> Evaluating on test set")
        elif opt.split.lower() == "train":
            whole_test_set = dataset.get_whole_training_set()
            print("===> Evaluating on train set")
        elif opt.split.lower() == "val":
            whole_test_set = dataset.get_whole_val_set()
            print("===> Evaluating on val set")
        else:
            raise ValueError("Unknown dataset split: " + opt.split)
        
        print("====> Query count:", whole_test_set.num_queries)

    elif opt.mode.lower() == "cluster":
        whole_train_set = dataset.get_whole_training_set(onlyDB=True)

    print("===> Building model")

    pretrained = not opt.fromscratch
    
    if opt.arch.lower() == "alexnet":
        encoder_dim = 256
        encoder = models.alexnet(pretrained=pretrained)
        # capture only features and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

    elif opt.arch.lower() == "vgg16":
        encoder_dim = 512
        encoder = models.vgg16(pretrained=pretrained)
        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        if pretrained:
            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]: 
                for p in l.parameters():
                    p.requires_grad = False

    elif opt.arch.lower().startswith("resnet"):
        
        variant = int(opt.arch.lower()[len("resnet"):])

        encoder = getattr(models, opt.arch.lower())(pretrained=False)
        encoder_dim = 256 if variant in [18, 34] else 1024

        encoder.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        state_dict = torch.load(join("pretrained", f"{opt.arch.lower()}_gray.pth"))["model"]
        encoder.load_state_dict(state_dict)

        for name, child in encoder.named_children():
            # freeze layers before conv_3
            if name == "layer3":
                break
            for params in child.parameters():
                params.requires_grad = False

        # train only layer4 of resnet
        # remove conv5_x, freeze previous ones
        layers = list(encoder.children())[:-3]

    if opt.mode.lower() == "cluster" and not opt.vladv2:
        layers.append(L2Norm())

    encoder = nn.Sequential(*layers)
    model = nn.Module() 
    model.add_module("encoder", encoder)

    if opt.mode.lower() != "cluster":
        if opt.pooling.lower() == "netvlad":
            net_vlad = netvlad.NetVLAD(num_clusters=opt.num_clusters, dim=encoder_dim, vladv2=opt.vladv2)
            if not opt.resume: 
                if opt.mode.lower() == "train":
                    initcache = join(opt.dataPath, "centroids", opt.arch + "_" + train_set.dataset + "_" + str(opt.num_clusters) +"_desc_cen.hdf5")
                else:
                    initcache = join(opt.dataPath, "centroids", opt.arch + "_" + whole_test_set.dataset + "_" + str(opt.num_clusters) +"_desc_cen.hdf5")

                if not exists(initcache):
                    raise FileNotFoundError("Could not find clusters, please run with --mode=cluster before proceeding")

                with h5py.File(initcache, mode="r") as h5: 
                    clsts = h5.get("centroids")[...]
                    traindescs = h5.get("descriptors")[...]
                    net_vlad.init_params(clsts, traindescs) 
                    del clsts, traindescs

            model.add_module("pool", net_vlad)
        elif opt.pooling.lower() == "max":
            global_pool = nn.AdaptiveMaxPool2d((1,1))
            model.add_module("pool", nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        elif opt.pooling.lower() == "avg":
            global_pool = nn.AdaptiveAvgPool2d((1,1))
            model.add_module("pool", nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        else:
            raise ValueError("Unknown pooling type: " + opt.pooling)

    isParallel = False
    if opt.nGPU > 1 and torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
        if opt.mode.lower() != "cluster":
            model.pool = nn.DataParallel(model.pool)
        isParallel = True

    if not opt.resume:
        model = model.to(device)
    
    if opt.mode.lower() == "train":
        if opt.optim.upper() == "ADAM":
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr)#, betas=(0,0.9))
        elif opt.optim.upper() == "SGD":
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, 
                model.parameters()), lr=opt.lr,
                momentum=opt.momentum,
                weight_decay=opt.weightDecay)

            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lrStep, gamma=opt.lrGamma)
        else:
            raise ValueError("Unknown optimizer: " + opt.optim)

        # original paper/code doesn"t sqrt() the distances, we do, so sqrt() the margin, I think :D
        criterion = nn.TripletMarginLoss(margin=opt.margin**0.5, 
                p=2, reduction="sum").to(device)

    if opt.resume:

        if opt.ckpt.lower() == "latest":
            resume_ckpt = join(opt.resume, "checkpoints", "checkpoint.pth")
        elif opt.ckpt.lower() == "best":
            resume_ckpt = join(opt.resume, "checkpoints", "model_best.pth")
        elif opt.ckpt.lower() == "path":
            resume_ckpt = opt.ckptpath

        if isfile(resume_ckpt):
            
            print("=> loading checkpoint '{}'".format(resume_ckpt))

            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint["epoch"]
            best_metric = checkpoint["best_score"]

            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            
            # create new state dict
            # '-> prefix 'module.' has to be removed

            for k, v in state_dict.items():
                new_k = k.replace("module.", "")
                new_state_dict[new_k] = v

            state_dict = new_state_dict
            model.load_state_dict(state_dict)
            model = model.to(device)

            if opt.mode == "train":
                optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))

    if opt.mode.lower() == 'test':
        print('===> Running evaluation step')
        epoch = opt.start_epoch
        recalls = test(whole_test_set, epoch, write_tboard=False)
    elif opt.mode.lower() == 'cluster':
        print('===> Calculating descriptors and clusters')
        get_clusters(whole_train_set)
    elif opt.mode.lower() == 'train':
        print('===> Training model')
        writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime('%b%d_%H-%M-%S')+'_'+opt.arch+'_'+opt.pooling))

        # write checkpoints in logdir
        logdir = writer.file_writer.get_logdir()
        opt.save_path = join(logdir, opt.save_path)
        if not opt.resume:
            makedirs(opt.save_path)

        with open(join(opt.save_path, 'flags.json'), 'w') as f:
            f.write(json.dumps(
                {k:v for k,v in vars(opt).items()}
                ))
        print('===> Saving state to:', logdir)

        not_improved = 0
        best_score = 0
        for epoch in range(opt.start_epoch+1, opt.nEpochs + 1):
            if opt.optim.upper() == 'SGD':
                scheduler.step(epoch)
            train(epoch)
            if (epoch % opt.evalEvery) == 0:
                recalls = test(whole_test_set, epoch, write_tboard=True)
                is_best = recalls[5] > best_score 
                if is_best:
                    not_improved = 0
                    best_score = recalls[5]
                else: 
                    not_improved += 1

                save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'recalls': recalls,
                        'best_score': best_score,
                        'optimizer' : optimizer.state_dict(),
                        'parallel' : isParallel,
                }, is_best)

                if opt.patience > 0 and not_improved > (opt.patience / opt.evalEvery):
                    print('Performance did not improve for', opt.patience, 'epochs. Stopping.')
                    break

        print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
        writer.close()
