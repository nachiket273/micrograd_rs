use crate::Val;


pub struct Neuron {
    weights: Vec<Val>,
    bias: Val,
    use_relu: bool,
    use_tanh: bool
}

impl Neuron {
    pub fn new(in_ch: u32, use_relu: bool, use_tanh: bool) -> Self {
        let weights: Vec<Val> = (0..in_ch)
        .map(|_| Val::from(rand::random::<f64>() * 2.0 - 1.0))
        .collect();

        let bias = Val::from(rand::random::<f64>() * 2.0 - 1.0);

        Neuron {
            weights,
            bias,
            use_relu,
            use_tanh
        }
    }

    pub fn forward(&self, x: Vec<Val>) -> Val {
        let sum: Val = self.weights
        .iter()
        .zip(x.iter())
        .map(|(weight, input)| {
            weight.clone() * input.clone()
        }).sum();

        let mut ret = sum + self.bias.clone();

        if self.use_relu {
            ret = ret.relu();
        }

        if self.use_tanh {
            ret = ret.tanh();
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
    pub fn new(in_ch: u32, out_ch: u32, use_relu: bool, use_tanh: bool) -> Self {
        let neurons = (0..out_ch)
        .map(|_| Neuron::new(in_ch, use_relu, use_tanh))
        .collect();

        Layer {
            neurons
        }        
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
    pub fn new(in_ch: u32, out_ch: Vec<u32>, use_relu: bool) -> Self {
        let mut layers = Vec::new();
        let mut in_ch_mut = in_ch;
        let sz = out_ch.len();

        for i in 0..sz {
            if i == sz-1 {
                layers.push(Layer::new(in_ch_mut, out_ch[i], false, true));
            }
            else {
                layers.push(Layer::new(in_ch_mut, out_ch[i], use_relu, false));
            }
            in_ch_mut = out_ch[i];
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