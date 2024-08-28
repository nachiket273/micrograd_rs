use std::{
    borrow::Borrow, fmt::Debug, hash::Hash, ops::Add
};

// Basic Implementation of struct Value
#[derive(Clone)]
struct Value {
    data: f64,
    grad: f64,
    operation: String,
    label: String
}

impl Value {
    fn new(data: f64, grad: f64,
        operation: String,
        label: String) -> Value {
            Value {
                data, grad, operation, label
            }
        }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
        && self.grad == other.grad
        && self.operation == other.operation
        && self.label == other.label
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.to_bits().hash(state);
        self.grad.to_bits().hash(state);
        self.operation.hash(state);
        self.label.hash(state);
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value")
        .field("data", &self.data)
        .field("grad", &self.grad)
        .field("operation", &self.operation)
        .field("label", &self.label)
        .finish()
    }
}

// Implementation of Add, Subtract, Multiply and Divide
impl Add<Value> for Value {
    type Output = Value;

    fn add(self, other: Value) -> Self::Output {
        add(&self, &other)
    }
}

fn add(a: &Value, b: &Value) -> Value {
    let addn = a.borrow().data + b.borrow().data;

    Value::new(addn, 0.0,
        "+".to_string(),
        "addition".to_string())
}


fn main() {
    let val = Value::new(10.0, 0.0, "".to_string(),
    "Val1".to_string());
    let val2 = Value::clone(&val);
    println!("Current struct: {:#?}", val);
    assert!(val.eq(&val2), "Not True");
}  