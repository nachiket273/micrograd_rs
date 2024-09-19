# micrograd_rs
[![Crate](https://img.shields.io/crates/v/micrograd_rs_nt.svg)](https://crates.io/crates/micrograd_rs_nt)


<br>

A Rust Beginner's implementation of [Andrej karpathy's micrograd](https://github.com/karpathy/micrograd). <br>
Here's [Link](https://www.youtube.com/watch?v=VMj-3S1tku0) to the YouTube video. <br>
Also thanks to [micrograd-rust](https://github.com/sloganking/micrograd-rust) for reference implementation in rust.
<br>



## The current implementation is limited to:<br>
==============================================<br>
* scalar valued autograd
* Implementation of forward and backward pass for arithmetic operations and power operation.
* Implementation of forward and backward pass for activation functions - tanh and relu.
<br>

## Sample Output: <br>
===============================================<br>
![Sample Training Output for MLP](./img/train.png)

<br>

## TO-DO:
==========================================
* graph visulization using libraries like [graphviz](https://graphviz.org/download/)
* Implementation of Tensor type library from scratch
* Improve implementation of MLP, activations etc for Tensors.


