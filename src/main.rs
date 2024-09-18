mod nn;
mod value;

use crate::value::Val;

fn main() {
    let mlp = nn::MLP::new(5, vec![3, 4, 1]);

}