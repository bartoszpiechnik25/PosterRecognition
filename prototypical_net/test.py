import torch
import pytest
from .model_prototype import PrototypicalNetwork


@pytest.fixture
def model():
    return PrototypicalNetwork()


def test_compute_prototypes(model):
    X = torch.randn(5, 3, 224, 224)
    cls_idx = torch.tensor([0, 1, 2, 3, 4])
    model.compute_prototypes(X, cls_idx)
    assert model.prototypes.shape == (4000, 768)


def test_forward(model):
    X = torch.randn(5, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 3, 4])
    preds, accuracy = model(X, y)
    assert preds.shape == (5,)
    assert accuracy is not None


def test_calculate_accuracy():
    predictions = torch.tensor([0, 1, 2, 3, 4])
    targets = torch.tensor([0, 1, 2, 3, 4])
    accuracy = PrototypicalNetwork.calculate_accuracy(predictions, targets)
    assert accuracy == 1.0


def test_euclidean_distance():
    x1 = torch.randn(5, 10)
    x2 = torch.randn(5, 10)
    distance = PrototypicalNetwork.euclidean_distance(x1, x2)
    assert distance.shape == (5,)


def test_euclidean_distance_with_different_shapes():
    x1 = torch.randn(32, 10)
    x2 = torch.randn(64, 10)
    distance = PrototypicalNetwork.euclidean_distance(x1, x2)
    assert distance.shape == (32, 64)


def test_euclidean_distance_with_one_dimension_as_1():
    x1 = torch.randn(1, 10)
    x2 = torch.randn(5, 10)
    distance = PrototypicalNetwork.euclidean_distance(x1, x2)
    assert distance.shape == (5,)