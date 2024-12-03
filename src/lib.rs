use burn::{module::Module, nn::{Linear, LinearConfig, Relu}, tensor::{backend::Backend, Shape, Tensor}};
use conv_block::{ConvBlock, ConvBlockMaxPool};

mod conv_block;
mod traning;
pub mod data;
pub mod circle_dataset;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    l1: ConvBlockMaxPool<B>,
    l2: ConvBlockMaxPool<B>,
    l3: ConvBlockMaxPool<B>,
    l4: ConvBlock<B>,
    l5: ConvBlock<B>,
    fc: FcBlock<B>,
    last: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn init(device: &B::Device) -> Model<B> {
        Model {
            l1: ConvBlockMaxPool::new([1, 32], [5, 5], device),
            l2: ConvBlockMaxPool::new([32, 64], [3, 3], device),
            l3: ConvBlockMaxPool::new([64, 128], [3, 3], device),
            l4: ConvBlock::new([128, 128], [3, 3], device),
            l5: ConvBlock::new([128, 4], [1, 1], device),
            fc: FcBlock::new(device),
            last: LinearConfig::new(16, 3).with_bias(true).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = input.dims();

        let x = input.reshape([batch_size, 1, height, width]).detach();
        assert_eq!(height, 200);
        assert_eq!(width, 200);

        let x = self.l1.forward(x);
        let x = self.l2.forward(x);
        let x = self.l3.forward(x);
        let x = self.l4.forward(x);
        let x = self.l5.forward(x);
        let Shape::<4> {
            dims: [batch_size, channel_size, height_size, width_size],
            ..
        } = x.shape();
        let x = x.reshape([batch_size, channel_size * height_size * width_size]);
        let x = self.fc.forward(x);
        self.last.forward(x)
    }

    // pub fn criterion(&self, images: Tensor<B, 3>, targets: Tensor<B, 2>) {
    //     let output = self.forward(images);
    //     let criterion1 = MseLoss::new().forward(output.clone(), targets.clone().div_scalar(200.0), Reduction::Mean);
    //     let criterion2 = output.sub(targets.div_scalar(200.0)).abs().mean();

    //     let loss = criterion1 + criterion2.mul_scalar(0.1);


    // }
}

#[derive(Module, Debug)]
struct FcBlock<B: Backend> {
    linear1: Linear<B>,
    act1: Relu,
    linear2: Linear<B>,
    act2: Relu,
}

impl<B: Backend> FcBlock<B> {
    pub fn new(device: &B::Device) -> Self {
        FcBlock {
            linear1: LinearConfig::new(4 * 21 * 21, 256)
                .with_bias(true).init(device),
            act1: Relu::new(),
            linear2: LinearConfig::new(256, 16)
                .with_bias(true).init(device),
            act2: Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = self.act1.forward(x);
        let x = self.linear2.forward(x);
        self.act2.forward(x)
    }
}


