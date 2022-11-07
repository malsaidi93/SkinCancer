import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--batch', type=int, default=16,
                        help='batch size')

    parser.add_argument('--model', type=str, default='resnet', help='model name')

    parser.add_argument('--num_classes', type=int, default=7, help="number \
                        of classes")

    parser.add_argument('--gpu', type=bool, default=True, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")

    parser.add_argument('--optimizer', type=str, default='adamx', help="type \
                        of optimizer")
    
    parser.add_argument('--imbalanced', type=bool, default=True, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")

    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')

    args = parser.parse_args()

    return args