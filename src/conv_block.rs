use burn::{module::Module, nn::{conv::{Conv2d, Conv2dConfig}, pool::{MaxPool2d, MaxPool2dConfig}, BatchNorm, BatchNormConfig, Relu}, prelude::Backend, tensor::Tensor};

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B, 2>,
    activation: Relu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(channels: [usize; 2], kernel_size: [usize; 2], device: &B::Device) -> Self {
        ConvBlock {
            conv: Conv2dConfig::new(channels, kernel_size).init(device),
            norm: BatchNormConfig::new(channels[1]).init(device),
            activation: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        self.activation.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct ConvBlockMaxPool<B: Backend> {
    conv: ConvBlock<B>,
    pool: MaxPool2d,
}

impl<B: Backend> ConvBlockMaxPool<B> {
    pub fn new(channels: [usize; 2], kernel_size: [usize; 2], device: &B::Device) -> Self {
        ConvBlockMaxPool {
            conv: ConvBlock::new(channels, kernel_size, device),
            pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        self.pool.forward(x)
    }
}

