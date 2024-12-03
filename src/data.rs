use burn::{data::dataloader::batcher::Batcher, prelude::Backend, tensor::{Tensor, TensorData}};

use crate::circle_dataset::CircleDatasetItem;

#[derive(Clone)]
pub struct CircleBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> CircleBatcher<B> {
    pub fn new(device: B::Device) -> CircleBatcher<B> {
        CircleBatcher { device }
    }
}

#[derive(Clone, Debug)]
pub struct CircleBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 2>,
}

impl<B: Backend> Batcher<CircleDatasetItem, CircleBatch<B>> for CircleBatcher<B> {
    fn batch(&self, items: Vec<CircleDatasetItem>) -> CircleBatch<B> {
        let image = items
            .iter()
            .map(|item| {
                TensorData::from(item.image).convert::<B::FloatElem>()
            })
            .map(|data| Tensor::<B, 2>::from_data(data, &self.device))
            .map(|tensor| tensor.reshape([1, 200, 200]))
            .map(|tensor| (tensor - 0.5)/0.5)
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 2>::from_data(
                    [[item.annotation.0, item.annotation.1, item.annotation.2]], 
                    &self.device,
                ).reshape([1, 3])
            })
            .collect();
        
        let images = Tensor::cat(image, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        CircleBatch { images, targets }
    }
}

