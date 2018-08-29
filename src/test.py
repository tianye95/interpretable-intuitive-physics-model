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
    num_feature, trainlabels = 306, ["mass", "force", "friction"]
    passthrough_dict = utils.get_passthrough(trainlabels, parameter_length=25, vector_length=num_feature)
    train_loader, shape_test_loader, parameter_test_loader = \
        dataset.getloader(args, labels=trainlabels, inframe=[0, 1, 2, 3], outframe=[4])

    # ------------------------- initialize model and optimizer -------------------------
    model = net.PhysicsModel(num_feature=num_feature, passthrough=passthrough_dict)
    if args.cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    start_epoch, best_epoch_error, model, optimizer = utils.load_best_checkpoint(args.resume, model, optimizer)

    # ------------------------- define criterion -------------------------
    criterion = evaluation.pixel

    shape_error = test(shape_test_loader, model, criterion, start_epoch, optimizer, args=args)
    parameter_error = test(parameter_test_loader, model, criterion, start_epoch, optimizer, args=args)

    print('Time elapsed: {:.2f}s'.format(time.time() - start_time))


def test(batch_loader, model, criterion, epoch, optimizer, args=None):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    end_time = time.time()

    for batch_id, (batch_input, batch_output, indices_dict) in enumerate(batch_loader):
        data_time.update(time.time() - end_time)
        model_input = utils.combinebatchdata(batch_input, autograd=False, cuda=args.cuda)
        target = utils.combinebatchdata(batch_output, autograd=False, cuda=args.cuda)

        prediction, extraoutputs = model(model_input, indices_dict)
        loss = criterion(prediction, target[4])
        losses.update(loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        # optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        if batch_id % args.log_interval == 0:
            utils.print_training_status(epoch, batch_id, len(batch_loader), losses)

    print('Epoch: [{0}] Average Loss {loss.avg:.6f}; Batch Avg Time {b_time.avg:.6f}'
          .format(epoch, loss=losses, b_time=batch_time))

    return losses.avg


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
    parser.add_argument('--project-name', default='test')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-8, 1e-2]), default=1e-6, metavar='LR',
                        help='Initial learning rate [1e-5, 1e-2] (default: 1e-3)')
    parser.add_argument('--prefetch', type=int, default=4)
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
