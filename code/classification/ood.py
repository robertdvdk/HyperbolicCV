# -----------------------------------------------------
# Change working directory to parent HyperbolicCV/code
import os
import sys
working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
os.chdir(working_dir)

lib_path = os.path.join(working_dir)
sys.path.append(lib_path)
# -----------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import configargparse

from openood.evaluation_api import Evaluator
# Reuse your existing initialization functions
from utils.initialize import select_model


class OpenOODWrapper(nn.Module):
    """Wrapper to make your ResNet compatible with OpenOOD evaluator"""

    def __init__(self, resnet_model):
        super().__init__()
        self.model = resnet_model

        # if self.model.predictor is None:
        #     raise ValueError("Model must have a predictor layer for OpenOOD evaluation")

        # self.feature_dim = self.model.embed_dim * self.model.block.expansion

    def forward(self, x, return_feature=False, return_feature_list=False):
        """Modified forward pass that can return features"""
        # # Forward through conv layers
        # out = self.model.conv1(x)
        # out_1 = self.model.conv2_x(out)
        # out_2 = self.model.conv3_x(out_1)
        # out_3 = self.model.conv4_x(out_2)
        # out_4 = self.model.conv5_x(out_3)
        # out = self.model.avg_pool(out_4)

        # # Flatten features
        # features = out.view(out.size(0), -1)

        # # Get logits
        logits = self.model.forward(x)

        # Return based on flags
        # if return_feature:
        #     return logits, features
        # elif return_feature_list:
        #     return logits, [features]
        return logits

    # def get_fc(self):
    #     """Return FC layer weights and bias as numpy arrays"""
    #     predictor = self.model.predictor

    #     # Handle Euclidean case (standard nn.Linear)
    #     if isinstance(predictor, nn.Linear):
    #         weight = predictor.weight.detach().cpu().numpy()
    #         bias = predictor.bias.detach().cpu().numpy() if predictor.bias is not None else np.zeros(predictor.out_features)
    #         return weight, bias

    #     # Handle Lorentz case (LorentzMLR)
    #     else:
    #         # For LorentzMLR, we need to extract the underlying parameters
    #         # Check what attributes your LorentzMLR has and adjust accordingly
    #         if hasattr(predictor, 'weight'):
    #             weight = predictor.weight.detach().cpu().numpy()
    #             bias = predictor.bias.detach().cpu().numpy() if hasattr(predictor, 'bias') and predictor.bias is not None else np.zeros(predictor.num_classes)
    #             return weight, bias
    #         else:
    #             # Fallback: create dummy parameters (works for MSP, Energy, ODIN)
    #             print("Warning: Using dummy FC parameters for Lorentz model. Some postprocessors may not work optimally.")
    #             num_classes = getattr(predictor, 'num_classes', 100)  # adjust default
    #             weight = np.random.randn(num_classes, self.feature_dim + 1) * 0.01
    #             bias = np.zeros(num_classes)
    #             return weight, bias

    # def get_fc_layer(self):
    #     """Return the predictor layer module"""
    #     return self.model.predictor


def getArguments():
    """Parse command-line options for testing"""
    parser = configargparse.ArgumentParser(description='OpenOOD Evaluation', add_help=True)

    # Required: checkpoint to evaluate
    parser.add_argument('--checkpoint', required=True, type=str,
                        help="Path to trained model checkpoint.")

    # OpenOOD settings
    parser.add_argument('--id_name', default='cifar100', type=str,
                        choices=['cifar10', 'cifar100', 'imagenet200', 'imagenet1k'],
                        help="ID dataset name for OpenOOD.")
    parser.add_argument('--data_root', default='./data', type=str,
                        help="Root directory for OpenOOD datasets.")
    parser.add_argument('--postprocessor', default='msp', type=str,
                        help="OOD detection method (e.g., msp, energy, odin, vim, knn, react, ash).")
    parser.add_argument('--batch_size', default=200, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument('--device', default='cuda:0', type=str,
                        help="Device to use for evaluation.")

    # Multiple postprocessors
    parser.add_argument('--test_all', action='store_true',
                        help="Test multiple postprocessors.")
    parser.add_argument("--init_method", type=str, choices=["old", "eye05", "eye1"])

    # Layer implementation settings
    parser.add_argument('--linear_method', default='theirs', type=str, choices=['ours', 'theirs'],
                        help="Select LorentzFullyConnected implementation: 'ours' (custom) or 'theirs' (Chen et al. 2022)")
    parser.add_argument('--batchnorm', default='default', type=str, choices=['default', 'train'],
                        help="Select LorentzBatchNorm behavior: 'default' (respect training mode) or 'train' (always use training branch)")

    args = parser.parse_args()
    return args


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from your training checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    print(os.listdir("./classification/output"))
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract training args (stored in checkpoint)
    train_args = checkpoint['args']

    print("Reconstructing model with training configuration:")
    print(f"  Dataset: {train_args.dataset}")
    print(f"  Encoder manifold: {train_args.encoder_manifold}")
    print(f"  Decoder manifold: {train_args.decoder_manifold}")
    print(f"  Num layers: {train_args.num_layers}")
    print(f"  Embedding dim: {train_args.embedding_dim}")

    # Determine image dimensions and num_classes from dataset
    if train_args.dataset == 'CIFAR-10':
        img_dim = [3, 32, 32]
        num_classes = 10
    elif train_args.dataset == 'CIFAR-100':
        img_dim = [3, 32, 32]
        num_classes = 100
    elif train_args.dataset == 'Tiny-ImageNet':
        img_dim = [3, 64, 64]
        num_classes = 200
    elif train_args.dataset == 'MNIST':
        img_dim = [1, 28, 28]
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {train_args.dataset}")

    # Recreate model using your existing select_model function
    train_args.linear_method = args.linear_method
    train_args.batchnorm = args.batchnorm
    print(f"Layer config: linear_method={args.linear_method}, batchnorm={args.batchnorm}")

    model = select_model(img_dim, num_classes, train_args)

    # Load weights
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    print(f"Model loaded successfully (trained for {checkpoint['epoch']+1} epochs)")

    return model, train_args


def main(args):
    device = args.device
    torch.cuda.set_device(device)

    # Load your trained model
    model, train_args = load_model_from_checkpoint(args.checkpoint, device)

    # Wrap for OpenOOD
    wrapped_model = OpenOODWrapper(model)
    wrapped_model.eval()

    print(f"\n{'='*60}")
    print("Starting OpenOOD Evaluation")
    print(f"{'='*60}")

    if args.test_all:
        # Test multiple postprocessors
        postprocessors = ['msp', 'energy', 'odin', 'vim', 'knn']
        print(f"Testing postprocessors: {postprocessors}\n")

        results = {}
        for method in postprocessors:
            print(f"\n{'-'*60}")
            print(f"Testing: {method.upper()}")
            print(f"{'-'*60}")

            try:
                evaluator = Evaluator(
                    wrapped_model,
                    id_name=args.id_name,
                    data_root=args.data_root,
                    postprocessor_name=method,
                    batch_size=args.batch_size,
                )

                # Evaluate OOD detection
                ood_metrics = evaluator.eval_ood(fsood=False, progress=True)
                results[method] = ood_metrics

                print(f"\nResults for {method.upper()}:")
                print(ood_metrics)

            except Exception as e:
                print(f"Error with {method}: {e}")
                results[method] = None

        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY - All Methods")
        print(f"{'='*60}")
        for method, metrics in results.items():
            if metrics is not None:
                print(f"\n{method.upper()}:")
                print(metrics)

    else:
        # Test single postprocessor
        print(f"Testing postprocessor: {args.postprocessor}\n")

        evaluator = Evaluator(
            wrapped_model,
            id_name=args.id_name,
            data_root=args.data_root,
            postprocessor_name=args.postprocessor,
            batch_size=args.batch_size,
        )

        # Evaluate ID accuracy
        print("Evaluating ID accuracy...")
        id_acc = evaluator.eval_acc(data_name='id')
        print(f"ID Accuracy: {id_acc:.2f}%")

        # Evaluate OOD detection
        print("\nEvaluating OOD detection...")
        ood_metrics = evaluator.eval_ood(fsood=False, progress=True)

        print(f"\n{'='*60}")
        print(f"Results for {args.postprocessor.upper()}")
        print(f"{'='*60}")
        print(ood_metrics)

    print(f"\n{'='*60}")
    print("Evaluation Complete")
    print(f"{'='*60}")


if __name__ == '__main__':
    args = getArguments()

    # Check OpenOOD is installed
    try:
        from openood.evaluation_api import Evaluator
    except ImportError:
        print("ERROR: OpenOOD not installed!")
        print("Please run: pip install git+https://github.com/Jingkang50/OpenOOD")
        print("            pip install libmr")
        exit(1)

    main(args)