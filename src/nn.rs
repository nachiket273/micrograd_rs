use crate::{value::Value, Val};

pub struct Neuron {
    weights: Vec<Val>,
    bias: Val,
    use_relu: bool
}

impl Neuron {
    pub fn new(in_ch: u32, use_relu: bool) -> Self {
        let weights: Vec<Val> = (0..in_ch)
        .map(|_| Val::from(rand::random::<f64>() * 2.0 - 1.0))
        .collect();

        let bias = Val::from(rand::random::<f64>() * 2.0 - 1.0);

        Neuron {
            weights,
            bias,
            use_relu
        }
    }

    pub fn forward(&self, x: Vec<Val>) -> Val {
        let mut sum: f64 = 0.0;

        for (weight, ip) in self.weights.iter().zip(x.iter()) {
            sum += weight.borrow_mut().data * ip.borrow_mut().data
        }

        let mut new_val = Val::from(sum);
        new_val.borrow_mut().operation = String::from("+");
        new_val.borrow_mut().backward = Some(|val: &Value |{
            for v in val.previous.iter() {
                v.borrow_mut().grad += val.grad;
            }
        });

        let mut ret = new_val + self.bias.clone();

        if self.use_relu {
            ret = ret.relu();
        }

        ret
    }

    pub fn parameters(&self) -> Vec<Val> {
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    }

    pub fn zero_grad(&self) {
        for param in self.parameters().iter() {
            param.borrow_mut().grad = 0.0;
        }
    }
}

pub struct Layer {
    neurons : Vec<Neuron>
}

impl Layer {
    pub fn new(in_ch: u32, out_ch: u32) -> Self {
        (0..out_ch)
        .map(|_| Neuron::new(in_ch))
        .collect()      
    }

    pub fn forward(&self, x: Vec<Val>) -> Vec<Val> {
        self.neurons
        .iter()
        .map(|neuron| neuron.forward(x.clone()))
        .collect()
    }

    pub fn parameters(&self) -> Vec<Val> {
        self.neurons
        .iter()
        .flat_map(|neuron| neuron.parameters())
        .collect()
    }

    pub fn zero_grad(&self) {
        for param in self.parameters().iter() {
            param.borrow_mut().grad = 0.0;
        }
    }
}

pub struct MLP {
    layers : Vec<Layer>
}

impl MLP {
    pub fn new(in_ch: u32, out_ch: Vec<u32>) -> Self {
        let mut layers = Vec::new();

        let mut in_ch_mut = in_ch;

        for out in out_ch {
            layers.push(Layer::new(in_ch_mut, out));
            in_ch_mut = out;
        }

        MLP {layers}
    }

    pub fn forward(&self, inputs: Vec<Val>) -> Vec<Val> {
        let mut out = inputs;
        for layer in self.layers.iter() {
            out = layer.forward(out);
        }
        out
    }

    pub fn parameters(&self) -> Vec<Val> {
        self.layers
        .iter()
        .flat_map(|layer| layer.parameters())
        .collect()
    }

    pub fn zero_grad(&self) {
        for param in self.parameters().iter() {
            param.borrow_mut().grad = 0.0;
        }
    }
}