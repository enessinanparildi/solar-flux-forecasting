import torch
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer
from tft_model import get_dataloaders


if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader, training_dataset = get_dataloaders()

    model = TemporalFusionTransformer.load_from_checkpoint(
        "D:/perceptivespace/lightning_logs/version_191/checkpoints/epoch=59-step=9120.ckpt"
    )
    model.eval()
    model = model.cpu()

    batch = next(iter(val_dataloader))
    x, y = batch

    keys = list(x.keys())

    class TFTExportWrapper(pl.LightningModule):
        def __init__(self, model, keys):
            super().__init__()
            self.model = model
            self.keys = keys

        def forward(self, *tensors):
            x = {k: v for k, v in zip(self.keys, tensors)}
            return self.model(x)


    tensors = tuple(x[k] for k in keys)
    wrapper = TFTExportWrapper(model, keys)


    batch_dim = torch.export.Dim("batch_size")

    wrapper.to_onnx(
        "models/tft_model_lighting.onnx",
        input_sample=tensors,
        export_params=True,
        opset_version=18,
        input_names=keys,
        output_names=["output"],
        dynamic_shapes={"tensors": tuple({0: batch_dim} for _ in keys)},
        dynamo=True,
    )

    print("Exported to tft_model.onnx")