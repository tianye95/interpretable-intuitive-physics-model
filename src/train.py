import os
import argparse
import time
import numpy as np
import torch
import torch.autograd
import matplotlib
matplotlib.use('Agg')
import path
import net
import utils
import dataset
import evaluation

# two labels:
# ["mass", "friction"], ["mass", "force"], ["force", "friction"],
# ["force", "mass"], ["friction", "mass"], ["friction", "force"]
# trainlabels = ["mass", "friction"]

# one label:
# ["mass"], ["friction"], ["force"]
# trainlabels = ["mass"]


def main(args):
    np.random.seed(0)
    torch.manual_seed(0)
    start_time = time.time()
    utils.create_image_records(args.visualization_path)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # ------------------------- get data loaders -------------------------
    num_feature = 306
    trainlabels = ["mass", "force", "friction"] if args.baseline else args.train_labels
    passthrough_dict = utils.get_passthrough(trainlabels, parameter_length=25, vector_length=num_feature)
    train_loader, shape_test_loader, parameter_test_loader = \
        dataset.getloader(args, labels=trainlabels, inframe=[0, 1, 2, 3], outframe=[4])

    # ------------------------- initialize model and optimizer -------------------------
    select = False if args.baseline else True
    model = net.PhysicsModel(num_feature=num_feature, passthrough=passthrough_dict, select=select)
    if args.cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    start_epoch, best_epoch_error, model, optimizer = utils.load_best_checkpoint(args.resume, model, optimizer)

    # ------------------------- define criterion -------------------------
    criterion = evaluation.image

    for epoch in range(args.start_epoch, args.epochs):
        print(trainlabels)
        epoch_error = train(train_loader, model, criterion, optimizer, epoch, passthrough_dict, args=args)
        is_best = epoch_error < best_epoch_error
        best_epoch_error = min(epoch_error, best_epoch_error)
        utils.save_checkpoint({'epoch': epoch + 1, 'best_epoch_error': best_epoch_error,
                               'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                              is_best=is_best, checkpoint_path=args.resume, epoch=epoch)

    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


def train(batch_loader, model, criterion, optimizer, epoch, passthrough_dict, args=None):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    otherlosses = {'feature': utils.AverageMeter(), 'image': utils.AverageMeter()}
    for label in passthrough_dict.keys():
        otherlosses[label] = utils.AverageMeter()
    end_time = time.time()

    for batch_id, (batch_input, batch_output, indices_dict) in enumerate(batch_loader):
        data_time.update(time.time() - end_time)
        model_input = utils.combinebatchdata(batch_input, autograd=True, cuda=args.cuda)
        target = utils.combinebatchdata(batch_output, autograd=False, cuda=args.cuda)
        loss_weight = utils.get_weight(target, cuda=args.cuda)

        prediction, extraoutputs = model(model_input, indices_dict)
        imageloss = criterion(prediction, target[4], loss_weight[4])
        otherlosses['image'].update(imageloss.data[0])

        featureloss = evaluation.feature(extraoutputs, passthrough_dict, weight=100)
        otherlosses['feature'].update(featureloss.data[0])

        changelosses = evaluation.change(extraoutputs, indices_dict, passthrough_dict, weight=100)
        for label, changeloss in changelosses.items():
            otherlosses[label].update(changeloss.data[0])

        loss = imageloss.clone()
        losses.update(loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        if batch_id % args.log_interval == 0:
            utils.print_training_status(epoch, batch_id, len(batch_loader), losses, otherlosses)

    print('Epoch: [{0}] Average Loss {loss.avg:.6f}; Batch Avg Time {b_time.avg:.6f}'
          .format(epoch, loss=otherlosses['image'], b_time=batch_time))

    return otherlosses['image'].avg


def parse_arguments():
    def restricted_float(x, inter):
        x = float(x)
        if x < inter[0] or x > inter[1]:
            raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
        return x

    paths = path.Paths()

    parser = argparse.ArgumentParser(description='collision')
    parser.add_argument('--image-height', default=256, help='height of resized image')
    parser.add_argument('--image-width', default=256, help='width of resized image')
    parser.add_argument('--output-height', default=256, help='height of output image')
    parser.add_argument('--output-width', default=256, help='width of output image')

    parser.add_argument('--info-path', default=paths.info_path, help='path to data information')
    parser.add_argument('--data-path', default=paths.data_path, help='path to image input and target')
    parser.add_argument('--visualization-path', default=os.path.join(paths.visualization_path))
    parser.add_argument('--resume', default=os.path.join(paths.model_path))
    parser.add_argument('--project-name', default='default')
    parser.add_argument('--train-labels', default=["mass", "force", "friction"])

    parser.add_argument('--baseline', action='store_true', default=False,
                        help='train the predictive model without disentangled representation')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-8, 1e-2]), default=1e-6, metavar='LR',
                        help='Initial learning rate [1e-5, 1e-2] (default: 1e-3)')
    parser.add_argument('--prefetch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--start-epoch', type=int, default=0, metavar='N',
                        help='Index of epoch to start (default: 0)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='How many batches to wait before logging training status')
    args = parser.parse_args()
    args.visualization_path = os.path.join(args.visualization_path, args.project_name)
    args.resume = os.path.join(args.resume, args.project_name)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
