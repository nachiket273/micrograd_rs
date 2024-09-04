use std::{
    fmt::Debug,
    hash::Hash,
    ops::{Add, Mul, Sub}
};
use uuid::Uuid;

// Struct to hold data, gradient, operation,
// backward function and previous node.
#[derive(Clone)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
    pub operation: String,
    pub backward: Option<fn(val: &mut Value)>,
    pub previous: Vec<Value>,
    pub id: Uuid
}

impl Value {
    fn new(data: f64) -> Value {
        Value {
            data,
            grad: 0.0,
            operation: "".to_string(),
            backward: None,
            previous: Vec::new(),
            id: Uuid::new_v4()
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Value{}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value")
        .field("data", &self.data)
        .field("grad", &self.grad)
        .field("operation", &self.operation)
        .finish()
    }
}


// Implementation of Add, Subtract, Multiply and Divide
impl Add for Value {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut add_val = Value::new(self.data + other.data);
        add_val.operation = String::from("+");
        add_val.previous = vec![self, other];
        add_val.backward = Some(|val: &mut Value| {
            val.previous[0].grad += val.grad;
            val.previous[1].grad += val.grad;
        });
        return add_val;
    }
}

impl Sub for Value {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut sub_val = Value::new(self.data - other.data);
        sub_val.operation = String::from("-");
        sub_val.previous = vec![self, other];
        sub_val.backward = Some(|val: &mut Value| {
            val.previous[0].grad += val.grad;
            val.previous[1].grad -= val.grad;
        });
        return sub_val;
    }
}

impl Mul for Value {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut mul_val = Value::new(self.data * other.data);
        mul_val.operation = String::from("*");
        mul_val.previous = vec![self, other];
        mul_val.backward = Some(|val: &mut Value| {
            val.previous[0].grad += val.grad * val.previous[1].grad;
            val.previous[1].grad += val.grad * val.previous[0].grad;
        });
        return mul_val;
    }
}

fn main() {
    let val = Value::new(10.0);
    let val2 = Value::clone(&val);
    println!("Value1: {:#?}", val);
    println!("Value2: {:#?}", val2);
    let val3 = val + val2;
    println!("Addition Value: {:#?}", val3);
}  