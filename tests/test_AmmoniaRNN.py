import torch
import pytest
from nh3pred.model_def import AmmoniaRNN

# --- Fixture to create a model instance ---
@pytest.fixture
def model():
    """Create and return an instance of the AmmoniaRNN model."""
    return AmmoniaRNN()


def test_model_initialization(model):
    """Check that the model can be created without errors."""
    assert isinstance(model, AmmoniaRNN)


def test_embedding_dimensions(model):
    """Check that embedding dimensions and counts are correctly defined."""
    expected_num_embeddings = [5, 3, 2]
    expected_embedding_dims = [10, 9, 8]

    for i, emb_layer in enumerate(model.embeddings):
        assert emb_layer.num_embeddings == expected_num_embeddings[i]
        assert emb_layer.embedding_dim == expected_embedding_dims[i]


def test_forward_output_shape(model):
    """Verify that the forward pass returns output of the correct shape."""
    batch_size = 2
    seq_len = 4

    # Create dummy input data
    x_continuous = torch.randn(seq_len, batch_size, 13 - 3)  # 13 total inputs - 3 categorical
    x_categoricals = [
        torch.randint(0, 5, (seq_len, batch_size)),
        torch.randint(0, 3, (seq_len, batch_size)),
        torch.randint(0, 2, (seq_len, batch_size)),
    ]

    # Forward pass
    output = model((x_continuous, x_categoricals))

    # Check that the output has the expected shape
    assert output.shape == (seq_len, batch_size, 1)


def test_backward_pass(model):
    """Check that the backward pass runs without errors"""
    batch_size = 2
    seq_len = 5
    x_continuous = torch.randn(seq_len, batch_size, 10)
    x_cat1 = torch.randint(0, 5, (seq_len, batch_size))
    x_cat2 = torch.randint(0, 3, (seq_len, batch_size))
    x_cat3 = torch.randint(0, 2, (seq_len, batch_size))

    output = model((x_continuous, [x_cat1, x_cat2, x_cat3]))
    loss = output.mean()
    loss.backward()  # doit se passer sans erreur


def test_invalid_input_dimension(model):
    """Test that an invalid input raises an error."""
    x_continuous = torch.randn(5, 2, 8)  # Mauvaise dimension
    x_cat1 = torch.randint(0, 5, (5, 2))
    x_cat2 = torch.randint(0, 3, (5, 2))
    x_cat3 = torch.randint(0, 2, (5, 2))

    with pytest.raises(RuntimeError):
        model((x_continuous, [x_cat1, x_cat2, x_cat3]))


def test_forward_no_nan(model):
    """Ensure that the model does not produce NaN values."""
    batch_size = 2
    seq_len = 3

    x_continuous = torch.randn(seq_len, batch_size, 13 - 3)
    x_categoricals = [
        torch.randint(0, 5, (seq_len, batch_size)),
        torch.randint(0, 3, (seq_len, batch_size)),
        torch.randint(0, 2, (seq_len, batch_size)),
    ]

    output = model((x_continuous, x_categoricals))
    assert not torch.isnan(output).any()
