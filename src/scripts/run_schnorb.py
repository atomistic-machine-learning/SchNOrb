#!/usr/bin/env python
import argparse
import logging
import os
import sys
from shutil import copyfile, rmtree

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import schnetpack as spk
import schnorb as stl
from schnetpack.atomistic import AtomisticModel
from schnetpack.nn.cutoff import HardCutoff, CosineCutoff, MollifierCutoff
from schnetpack.train import HeatmapMAE, MeanAbsoluteError, RootMeanSquaredError
from schnetpack.utils import read_from_json
from schnorb import SchNOrbProperties
from schnorb.data import SchNOrbAtomsData
from schnorb.rotations import OrcaRotator, AimsRotator

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
np.set_printoptions(linewidth=120, precision=3, suppress=True)


def get_parser():
    """ Setup parser for command line arguments """
    main_parser = argparse.ArgumentParser()

    ## command-specific
    cmd_parser = argparse.ArgumentParser(add_help=False)
    cmd_parser.add_argument('--cuda', help='Set flag to use GPU',
                            action='store_true')
    cmd_parser.add_argument('--logger',
                            help='Choose how to log training process',
                            choices=['csv', 'tensorboard'], default='csv')
    cmd_parser.add_argument('--parallel', help='Run on all available GPUs',
                            action='store_true')
    cmd_parser.add_argument('--batch_size', type=int,
                            help='Number of validation scripts',
                            default=32)

    ## training
    train_parser = argparse.ArgumentParser(add_help=False,
                                           parents=[cmd_parser])
    train_parser.add_argument('datapath', help='Path to ASE DB')
    train_parser.add_argument('modelpath',
                              help='Path / destination to (re)store model')
    train_parser.add_argument('--seed', type=int, default=None,
                              help='Set random seed for torch and numpy.')
    train_parser.add_argument('--overwrite', help='Remove previous results.',
                              action='store_true')

    train_parser.add_argument('--rndrot', help='Apply random rotations.',
                              action='store_true')
    train_parser.add_argument('--forces', help='Train/predict forces',
                              action='store_true')
    train_parser.add_argument('--aims', help='AIMS',
                              action='store_true')

    train_parser.add_argument('--sgdr', help='SGD with warm restarts.',
                              action='store_true')
    train_parser.add_argument('--t0', type=int,
                              help='Length of initial restart episode',
                              default=30)
    train_parser.add_argument('--tmult', type=int,
                              help='Multiplier of restart episode length',
                              default=1)
    train_parser.add_argument('--tpat', type=int, help='Restart patience',
                              default=5)

    # data split
    train_parser.add_argument('--split_path',
                              help='Path / destination of npz with data splits',
                              default=None)
    train_parser.add_argument('--split',
                              help='Split into [train] [validation] and remaining for testing',
                              type=int, nargs=2, default=[None, None])
    train_parser.add_argument('--max_epochs', type=int,
                              help='Number of training epochs',
                              default=500000)
    train_parser.add_argument('--lr', type=float, help='Initial learning rate',
                              default=1e-4)
    train_parser.add_argument('--lr_patience', type=int,
                              help='Epochs without improvement before reducing the learning rate',
                              default=25)
    train_parser.add_argument('--lr_decay', type=float,
                              help='Learning rate decay',
                              default=0.5)
    train_parser.add_argument('--lr_min', type=float,
                              help='Learning rate decay',
                              default=1e-6)

    ## evaluation
    eval_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    eval_parser.add_argument('datapath', help='Path of QM9 dataset directory')
    eval_parser.add_argument('modelpath', help='Path of stored model')
    eval_parser.add_argument('--split',
                             help='Evaluate on trained model on given split',
                             choices=['train', 'validation', 'test'],
                             default=['test'], nargs='+')
    eval_parser.add_argument('--reference_data', type=str, nargs='+',
                             default=None,
                             help='Path to directories containing FHI aims ouput. Used once to initialize data set')

    # model-specific parsers
    model_parser = argparse.ArgumentParser(add_help=False)

    #######  SchNet  #######
    schnet_parser = argparse.ArgumentParser(add_help=False,
                                            parents=[model_parser])
    schnet_parser.add_argument('--orbbasis', type=int,
                               help='Feature dimension',
                               default=1000)
    schnet_parser.add_argument('--factors', type=int, help='Feature dimension',
                               default=1000)
    schnet_parser.add_argument('--directions', type=int, help='Dim multiplier',
                               default=12)
    schnet_parser.add_argument('--pair_features', type=int,
                               help='Feature dimension',
                               default=100)
    schnet_parser.add_argument('--interactions', type=int,
                               help='Number of interaction blocks',
                               default=3)
    schnet_parser.add_argument('--lmax', type=int,
                               help='Max angular momentum',
                               default=2)
    schnet_parser.add_argument('--cutoff', type=float, default=10.)
    schnet_parser.add_argument(
        "--cutoff_function",
        help="Functional form of the cutoff",
        choices=["hard", "cosine", "mollifier"],
        default="cosine",
    )
    schnet_parser.add_argument('--minimal', action='store_true',
                               help='Just use one layer of Gaussians as descriptor. ' +
                                    'As a test for systems like O2.')
    schnet_parser.add_argument('--baseline', action='store_true',
                               help='Basic symmetric prediction used as baseline.')
    schnet_parser.add_argument('--quambo', action='store_true')
    schnet_parser.add_argument('--coupled', action='store_true')

    ## setup subparser structure
    cmd_subparsers = main_parser.add_subparsers(dest='mode',
                                                help='Command-specific arguments')
    cmd_subparsers.required = True
    subparser_train = cmd_subparsers.add_parser('train', help='Training help')
    subparser_eval = cmd_subparsers.add_parser('eval', help='Training help')
    subparser_pred = cmd_subparsers.add_parser('pred', help='Training help')

    train_subparsers = subparser_train.add_subparsers(dest='model',
                                                      help='Model-specific arguments')
    train_subparsers.required = True

    subparser_schnet_train = train_subparsers.add_parser('schnet',
                                                         help='SchNet help',
                                                         parents=[train_parser,
                                                                  schnet_parser])

    eval_subparsers = subparser_eval.add_subparsers(dest='model',
                                                    help='Model-specific arguments')
    subparser_schnet_eval = eval_subparsers.add_parser('schnet',
                                                       help='SchNet help',
                                                       parents=[eval_parser,
                                                                schnet_parser])

    pred_parser = argparse.ArgumentParser(add_help=False, parents=[cmd_parser])
    pred_parser.add_argument('datapath', help='Path of QM9 dataset directory')
    pred_parser.add_argument('modelpath', help='Path of stored model')
    pred_parser.add_argument('--split',
                             help='Evaluate on trained model on given split',
                             choices=['train', 'validation', 'test'],
                             default=None)
    pred_parser.add_argument('--limit', type=int, help='Maximum examples',
                             default=None)

    subparser_export = cmd_subparsers.add_parser('export', help='Export help')
    subparser_export.add_argument('datapath', help='Path of stored model')
    subparser_export.add_argument('modelpath', help='Path of stored model')
    subparser_export.add_argument('destpath',
                                  help='Destination path for exported model')
    subparser_export.add_argument('--batch_size', type=int,
                                  help='Number of validation scripts',
                                  default=100)

    pred_subparsers = subparser_pred.add_subparsers(dest='model',
                                                    help='Model-specific arguments')
    subparser_schnet_pred = pred_subparsers.add_parser('schnet',
                                                       help='SchNet help',
                                                       parents=[pred_parser,
                                                                schnet_parser])

    return main_parser


def train(args, model, train_loader, val_loader, device):
    # setup hook and logging
    hooks = [
        spk.train.MaxEpochHook(args.max_epochs)
    ]

    # filter for trainable parameters (https://github.com/pytorch/pytorch/issues/679)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=args.lr)

    schedule = spk.train.ReduceLROnPlateauHook(optimizer,
                                               patience=args.lr_patience,
                                               factor=args.lr_decay,
                                               min_lr=args.lr_min,
                                               window_length=1,
                                               stop_after_min=True)
    schedule.scheduler = ReduceLROnPlateau(
        optimizer,
        patience=schedule.patience,
        factor=schedule.factor,
        min_lr=schedule.min_lr,
    )
    hooks.append(schedule)

    metrics = [MeanAbsoluteError(SchNOrbProperties.ham_prop),
               RootMeanSquaredError(SchNOrbProperties.ham_prop),
               MeanAbsoluteError(SchNOrbProperties.ov_prop),
               RootMeanSquaredError(SchNOrbProperties.ov_prop),
               MeanAbsoluteError(SchNOrbProperties.en_prop),
               RootMeanSquaredError(SchNOrbProperties.en_prop)]

    if args.forces:
        metrics += [MeanAbsoluteError(SchNOrbProperties.f_prop),
                    RootMeanSquaredError(SchNOrbProperties.f_prop)]

    if args.logger == 'csv':
        logger = spk.train.CSVHook(os.path.join(args.modelpath, 'log'),
                                   metrics)
        hooks.append(logger)
    elif args.logger == 'tensorboard':
        metrics.append(HeatmapMAE(SchNOrbProperties.ham_prop))
        metrics.append(HeatmapMAE(SchNOrbProperties.ov_prop))

        logger = spk.train.TensorboardHook(os.path.join(args.modelpath, 'log'),
                                           metrics, log_histogram=True,
                                           img_every_n_epochs=100)
        hooks.append(logger)

    # setup loss function
    def loss(batch, result):
        diff = batch[SchNOrbProperties.ham_prop] - result[SchNOrbProperties.ham_prop]
        diff = diff ** 2
        err_ham = torch.mean(diff)

        diff = batch[SchNOrbProperties.ov_prop] - result[SchNOrbProperties.ov_prop]
        diff = diff ** 2
        err_overlap = torch.mean(diff)

        diff = batch[SchNOrbProperties.en_prop] - result[SchNOrbProperties.en_prop]
        diff = diff ** 2
        err_energy = torch.mean(diff)

        if args.forces:
            diff = batch[SchNOrbProperties.f_prop] - result[SchNOrbProperties.f_prop]
            diff = diff ** 2
            err_energy = 0.1 * err_energy + 0.9 * torch.mean(diff)

        err_sq = err_ham + err_overlap + err_energy

        if torch.sum(torch.isnan(err_sq)) > 0:
            print("NaN loss")
            return None

        return err_sq

    trainer = spk.train.Trainer(args.modelpath, model, loss, optimizer,
                                train_loader, val_loader, hooks=hooks)
    trainer.train(device)


def evaluate(args, model, train_loader, val_loader, test_loader, device):
    header = ['Subset', SchNOrbProperties.ham_prop + ' MAE',
              SchNOrbProperties.ham_prop + ' RMSE']

    metrics = [
        MeanAbsoluteError(SchNOrbProperties.ham_prop),
        RootMeanSquaredError(SchNOrbProperties.ham_prop)
    ]

    header += [SchNOrbProperties.ov_prop + ' MAE', SchNOrbProperties.ov_prop + ' RMSE']
    metrics += [
        MeanAbsoluteError(SchNOrbProperties.ov_prop),
        RootMeanSquaredError(SchNOrbProperties.ov_prop)
    ]

    header += [SchNOrbProperties.en_prop + ' MAE', SchNOrbProperties.en_prop + ' RMSE']
    metrics += [
        MeanAbsoluteError(SchNOrbProperties.en_prop),
        RootMeanSquaredError(SchNOrbProperties.en_prop)
    ]

    results = []
    if 'train' in args.split:
        results.append(['training'] + ['%.5f' % i for i in
                                       evaluate_dataset(metrics, model,
                                                        train_loader, device)])

    if 'validation' in args.split:
        results.append(['validation'] + ['%.5f' % i for i in
                                         evaluate_dataset(metrics, model,
                                                          val_loader, device)])

    if 'test' in args.split:
        results.append(['test'] + ['%.5f' % i for i in
                                   evaluate_dataset(metrics, model,
                                                    test_loader, device)])

    header = ','.join(header)
    results = np.array(results)

    np.savetxt(os.path.join(args.modelpath, 'evaluation.csv'), results,
               header=header, fmt='%s', delimiter=',')


def evaluate_dataset(metrics, model, loader, device):
    for metric in metrics:
        metric.reset()

    for batch in loader:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        result = model(batch)

        for metric in metrics:
            metric.add_batch(batch, result)

    results = [
        metric.aggregate() for metric in metrics
    ]
    return results


def predict_dataset(model, loader, device, limit=None):
    properties = [SchNOrbProperties.ham_prop, SchNOrbProperties.ov_prop,
                  SchNOrbProperties.en_prop]
    basic_entries = ['_positions', '_atomic_numbers']

    results = {
        p: [] for p in properties
    }

    reference = {
        d: [] for d in basic_entries + properties
    }

    count = 0

    for batch in tqdm(loader, ncols=100):
        for key in basic_entries:
            reference[key].append(batch[key].detach().numpy())

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }
        result = model(batch)

        Href = batch[SchNOrbProperties.ham_prop].cpu().detach().numpy()
        Sref = batch[SchNOrbProperties.ov_prop].cpu().detach().numpy()
        Eref = batch[SchNOrbProperties.en_prop].cpu().detach().numpy()
        Hpred = result[0].cpu().detach().numpy()
        Spred = result[1].cpu().detach().numpy()
        Epred = result[2].cpu().detach().numpy()

        results[SchNOrbProperties.ham_prop].append(Hpred)
        results[SchNOrbProperties.ov_prop].append(Spred)
        results[SchNOrbProperties.en_prop].append(Epred)

        reference[SchNOrbProperties.ham_prop].append(Href)
        reference[SchNOrbProperties.ov_prop].append(Sref)
        reference[SchNOrbProperties.en_prop].append(Eref)

        count += Epred.shape[0]
        if limit is not None and limit < count:
            break

    results = {
        p + '_pred': np.vstack(val) for p, val in results.items()
    }

    reference = {key: np.vstack(reference[key]) for key in
                 properties + basic_entries}

    return results, reference


def get_model(train_args, basisdef, orbital_energies, mean, stddev,
              parallelize):
    if train_args.cutoff_function == "hard":
        cutoff_network = HardCutoff
    elif train_args.cutoff_function == "cosine":
        cutoff_network = CosineCutoff
    elif train_args.cutoff_function == "mollifier":
        cutoff_network = MollifierCutoff

    num_gaussians = int(train_args.cutoff * 5)
    dirs = train_args.directions if train_args.directions > 0 else None

    schnorb = stl.model.SchNOrb(n_factors=train_args.factors,
                                lmax=train_args.lmax,
                                n_interactions=train_args.interactions,
                                directions=dirs,
                                n_cosine_basis=train_args.orbbasis,
                                n_gaussians=num_gaussians,
                                cutoff_network=cutoff_network,
                                coupled_interactions=args.coupled)
    print('SchNorb params: %.2fM' % (
            sum(p.numel() for p in schnorb.parameters()) / 1000000.0))
    hamiltonian = stl.model.Hamiltonian(basisdef,
                                        lmax=train_args.lmax,
                                        n_cosine_basis=train_args.orbbasis,
                                        directions=dirs,
                                        orbital_energies=orbital_energies,
                                        quambo=train_args.quambo,
                                        mean=mean, stddev=stddev,
                                        return_forces=train_args.forces,
                                        create_graph=train_args.forces)
    print('Outnet params: %.2fM' % (
            sum(p.numel() for p in hamiltonian.parameters()) / 1000000.0))
    schnorb = AtomisticModel(schnorb, hamiltonian)

    if parallelize:
        schnorb = nn.DataParallel(schnorb)

    return schnorb


def export_model(args, basisdef, orbital_energies, mean, stddev):
    jsonpath = os.path.join(args.modelpath, 'args.json')
    train_args = read_from_json(jsonpath)
    model = get_model(train_args, basisdef, orbital_energies, mean, stddev,
                      False)
    model.load_state_dict(
        torch.load(os.path.join(args.modelpath, 'best_model'),
                   map_location='cpu'))

    torch.save(model, args.destpath)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    argparse_dict = vars(args)
    jsonpath = os.path.join(args.modelpath, 'args.json')

    if args.mode == 'train':
        if args.overwrite and os.path.exists(args.modelpath):
            rmtree(args.modelpath)

        if not os.path.exists(args.modelpath):
            os.makedirs(args.modelpath)

        spk.utils.to_json(jsonpath, argparse_dict)
        spk.utils.set_random_seed(args.seed)
        train_args = args
    else:
        train_args = spk.utils.read_from_json(jsonpath)

    properties = [
        SchNOrbProperties.ham_prop,
        SchNOrbProperties.ov_prop,
        SchNOrbProperties.en_prop
    ]
    if train_args.forces:
        properties.append(SchNOrbProperties.f_prop)

    if train_args.aims:
        rot = AimsRotator
    else:
        rot = OrcaRotator

    hamiltonian_data = SchNOrbAtomsData(args.datapath,
                                        load_only=properties,
                                        add_rotations=train_args.rndrot,
                                        rotator_cls=rot)
    basisdef = hamiltonian_data.get_metadata('basisdef')
    basisdef = np.array(basisdef)
    orbital_energies = hamiltonian_data.get_metadata('orbital_energies')
    orbital_energies = None if basisdef is None else np.array(orbital_energies)

    # split data
    split_path = os.path.join(args.modelpath, 'split.npz')
    if args.mode == 'train':
        if args.split_path is not None:
            copyfile(args.split_path, split_path)

    data_train, data_val, data_test = hamiltonian_data.create_splits(
        *train_args.split, split_file=split_path)

    data_val.add_rotations = False
    data_test.add_rotations = False

    train_loader = spk.data.AtomsLoader(data_train, batch_size=args.batch_size,
                                        sampler=RandomSampler(data_train),
                                        num_workers=4, pin_memory=True)

    val_loader = spk.data.AtomsLoader(data_val, batch_size=args.batch_size,
                                      num_workers=4, pin_memory=True)

    if args.mode == 'train':
        mean, stddev = train_loader.get_statistics(SchNOrbProperties.en_prop, True)
        mean = mean[SchNOrbProperties.en_prop]
        stddev = stddev[SchNOrbProperties.en_prop]
    else:
        mean, stddev = None, None

    if args.mode == 'export':
        export_model(args, basisdef, orbital_energies, mean, stddev)
        sys.exit(0)

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.mode == 'eval' or args.mode == 'pred':
        model = torch.load(os.path.join(args.modelpath, 'best_model'))
    else:
        model = get_model(train_args, basisdef, orbital_energies,
                          mean, stddev, parallelize=args.parallel).to(device)

    print('Total params: %.2fM' % (
            sum(p.numel() for p in model.parameters()) / 1000000.0))

    if args.mode == 'train':
        logging.info("Training...")
        train(args, model, train_loader, val_loader, device)
    elif args.mode == 'eval':
        test_loader = spk.data.AtomsLoader(data_test,
                                           batch_size=args.batch_size,
                                           num_workers=2, pin_memory=True)
        evaluate(args, model, train_loader, val_loader, test_loader, device)
    elif args.mode == 'pred':
        if args.split is None:
            data_loader = spk.data.AtomsLoader(hamiltonian_data,
                                               batch_size=args.batch_size,
                                               num_workers=0, pin_memory=True)
        elif args.split == 'test':
            data_loader = spk.data.AtomsLoader(data_test,
                                               batch_size=args.batch_size,
                                               num_workers=2, pin_memory=True)
        elif args.split == 'validation':
            data_loader = val_loader
        elif args.split == 'train':
            data_loader = train_loader
        else:
            data_loader = None

        results, inputs = predict_dataset(model, data_loader, device,
                                          args.limit)
        predict_file = os.path.join(args.modelpath, 'prediction.npz')
        np.savez(predict_file, **results, **inputs)
    else:
        print('Unknown mode:', args.mode)
