import torch
from pytorch_forecasting import TemporalFusionTransformer
from tft_model import get_dataloaders


if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader, training_dataset = get_dataloaders()

    model = TemporalFusionTransformer.load_from_checkpoint(
        "D:/perceptivespace/lightning_logs/version_191/checkpoints/epoch=59-step=9120.ckpt"
    )
    model.eval()
    model.requires_grad_(False)
    model = model.cpu()


    # Get a real batch to trace with
    batch = next(iter(val_dataloader))
    x, y = batch
    # Wrapper converts positional tensor args back to the dict TFT expects
    class TFTExportWrapper(torch.nn.Module):
        def __init__(self, model, keys):
            super().__init__()
            self.model = model
            self.keys = keys

        def forward(self, *tensors):
            x = {k: v for k, v in zip(self.keys, tensors)}
            return self.model(x)

    keys = list(x.keys())
    tensors = tuple(x[k] for k in keys)

    def forward_fn(*tensors):
        inp = {k: v for k, v in zip(keys, tensors)}
        return model.forward(inp)

    with torch.no_grad():
        traced_fn = torch.jit.trace(forward_fn, tensors, strict=False)

    torch.onnx.export(
        traced_fn,
        tensors,
        "models/tft_model.onnx",
        opset_version=18,
        input_names=keys,
        output_names=["output"],
        dynamic_shapes={key: {0: "batch_size"} for key in keys},
        fallback=True,
    )

