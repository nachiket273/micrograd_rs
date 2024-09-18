use crate::Val;

pub struct Neuron {
    weights: Vec<Val>,
    bias: val,
    use_relu: bool
}

impl Neuron {
    pub fn new(in_ch: u32, use_relu: bool) -> Self {
        let weights: Vec<Val> = (0..in_ch)
        .map(|_| Val::new(Value::new(rand::random::<f64>() * 2.0 - 1.0)))
        .collect();

        let bias = rand::random::<f64>() * 2.0 - 1.0;

        Neuron {
            weights,
            bias,
            use_relu
        }
    }

    pub fn forward(&self, x: Vec<Val>) -> Val {
        let sum: Val = self.weights.iter()
        .zip(x.iter())
        .map(|weight, ip| {
            let part_sum = weight.clone() * ip.clone();
            part_sum
        })
        .sum();

        let ret = sum + self.bias.clone();

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
}

pub struct MLP {
    layers : Vec<Layer>
}

impl MLP {
    pub fn new(in_ch: u32, out_ch: Vec<u32>) -> Self {
        let mut layers = Vec::new();

        let mut in_ch_mut = in_ch;

        for out in out_ch {
            layers.push(Vec::new(in_ch_mut, out));
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

    pub fn zero_grad(&self) -> None {
        for param in self.parameters().iter() {
            param.borrow_mut().grad = 0.0;
        }
    }
}