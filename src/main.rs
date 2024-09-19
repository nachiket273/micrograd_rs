mod nn;
mod value;

use std::{env, collections::HashSet};

use rand::{distributions::Uniform, Rng};
use crate::value::Val;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    // Create dataset:
    // Inputs: Generate Input values between 0 to input_size * 2;
    let input_size: u32 = 5;
    let mut seen: HashSet<u64> = HashSet::new();
    let mut rng = rand::thread_rng();
    let range = Uniform::new(0, u64::from(input_size * 2));
    let mut inputs: Vec<Val> = Vec::new();
    let mut i = 0;
    while i < input_size {
        let candidate = rng.sample(&range);
        if !seen.contains(&candidate) {
            seen.insert(candidate);
            inputs.push(Val::from(candidate as f64));
            i += 1;
        }
    }
    println!("Dataset::");
    println!("Input: ");
    println!("{:#?}", inputs);

    //Targets: target is either 1 or -1
    // Let's set it to 1 if input is [0, input_size)
    // and -1 if input is [input_size, input_size *2)
    let mut targets: Vec<Val> = Vec::new();
    for val in inputs.iter() {
        if val.borrow_mut().data < input_size as f64 {
            targets.push(Val::from(1.0));
        }
        else {
            targets.push(Val::from(-1.0));
        }
    }
    println!("Targets: ");
    println!("{:#?}", targets);


    // Neural Network with 2 hidden layers
    let mlp = nn::MLP::new(input_size, vec![4, 4, 1], false);

    let params = mlp.parameters();
    println!("num params: {:#?}", params.len());

    // Training loop
    let step_size = 1e-2;
    let epochs = 10;
    println!("Training: ");
    for i in 0..epochs {
        mlp.zero_grad();

        let preds = mlp.forward(inputs.clone());

        // Calculate loss
        let losses: Vec<Val> = preds.iter()
        .zip(targets.iter())
        .map(|(pred, tgt)| (pred.clone() - tgt.clone()).pow(2.0))
        .collect();
        
        let tot_loss = losses.into_iter().sum::<Val>();

        println!("Epoch: {:#?}, loss: {:#?}", i+1, tot_loss.borrow().data);

        tot_loss.backward();

        // Update parameters
        let params = mlp.parameters();
        for param in params.iter() {
            let grad = param.borrow().grad;
            param.borrow_mut().data += step_size * -grad;
        }
    }
}