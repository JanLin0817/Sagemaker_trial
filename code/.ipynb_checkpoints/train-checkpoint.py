import os
import sys
import argparse
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from easy_net import Net
from dataloader import get_dataloader    


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
    
def parser():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--use-cuda', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    # Data, model, and output directories
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_INPUT_DIR', './data'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './checkpoint'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', None))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST', None))

    return parser.parse_known_args()
    
if __name__ == "__main__":
    args, unk_args = parser()
    dataroot = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    
    # for local training
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    # TODO: device
    
    # data loader
    loader_dict = get_dataloader(dataroot, args.batch_size)
    trainloader, testloader, classes = loader_dict['train'], loader_dict['test'], loader_dict['classes']

    # network
    net = Net()
    
    # loss
    criterion = nn.CrossEntropyLoss()
    
    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    # network
    net = Net()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for ii, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if ii % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, ii + 1, running_loss / 2000))
                running_loss = 0.0

#         logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#             loss, loss, len(trainloader.dataset),
#             100. * loss / len(trainloader.dataset)))
        logger.info('Test set: Average loss: {:.4f}'.format(loss))
        torch.save(net.state_dict(), f'{model_dir}/cifar_net_epoch_{epoch}.pth')
    
    log_args = str(args)
    log_args += f' output_dir exist: {os.path.isdir(output_dir)}'
    log_args += f' model_dir exist: {os.path.isdir(model_dir)}'
    with open(os.path.join(output_dir, 'log.txt'), 'w') as f:
        f.write(log_args)                
