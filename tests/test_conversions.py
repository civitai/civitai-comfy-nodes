import torch

from civitai_comfy_nodes import conversions


def test_image_tensor_data_url_round_trip():
    tensor = torch.rand(1, 32, 48, 3)
    data_url = conversions.image_tensor_to_data_url(tensor)
    assert data_url.startswith("data:image/png;base64,")

    import base64

    png = base64.b64decode(data_url.split(",", 1)[1])
    restored = conversions.bytes_to_image_tensor(png)
    assert restored.shape == (1, 32, 48, 3)
    assert torch.allclose(tensor, restored, atol=1 / 255)


def test_batch_to_data_urls():
    tensor = torch.rand(3, 16, 16, 3)
    urls = conversions.image_tensor_to_data_urls(tensor)
    assert len(urls) == 3
    assert all(u.startswith("data:image/png;base64,") for u in urls)


def test_stack_resizes_mismatched_images():
    a = torch.rand(1, 32, 32, 3)
    b = torch.rand(1, 64, 48, 3)
    stacked = conversions.stack_image_tensors([a, b])
    assert stacked.shape == (2, 32, 32, 3)
