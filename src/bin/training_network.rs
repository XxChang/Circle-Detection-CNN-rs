use burn::{backend::Wgpu, data::dataloader::DataLoaderBuilder, nn::loss::{MseLoss, Reduction}, optim::{AdamConfig, GradientsParams, Optimizer}, prelude::Backend, tensor::{cast::ToElement, Tensor}};
use circle_detection_cnn::{circle_dataset::CircleDataset, data::CircleBatcher};
use burn::backend::Autodiff;
use burn::backend::Wgpu
fn main() {
    let dataset = CircleDataset::read_train_csv("/home/xuchang/github/Circle-Detection-CNN/train_set.csv");
    let web_gpu = Default::default();

    let mut model = circle_detection_cnn::Model::init(&web_gpu);
    let mut optim = AdamConfig::new().init();

    let batcher_train = CircleBatcher::<Autodiff<Wgpu>>::new(web_gpu);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(64)
        .shuffle(42)
        .num_workers(4)
        .build(dataset);

    let num_epochs = 10;
    for epoch in 1..num_epochs + 1 {
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            let output = model.forward(batch.images);
            let criterion1 = MseLoss::new().forward(output.clone(), batch.targets.clone().div_scalar(200.0), Reduction::Mean);
            let criterion2 = output.clone().sub(batch.targets.clone().div_scalar(200.0)).abs().mean();
            let loss = criterion1.clone() + criterion2.mul_scalar(0.1);
            // let accuracy = accuracy(output, batch.targets);

            println!(
                "[Train - Epoch {} - Iteration {}] Loss {:.3} | Accuracy {:.3}", 
                epoch,
                iteration,
                loss.clone().into_scalar(),
                criterion1
            );

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(1e-4, model, grads);
        }
    }
}

