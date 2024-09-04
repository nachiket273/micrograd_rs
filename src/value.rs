use std::{
    fmt::Debug,
    hash::Hash,
    ops::{Add, Mul, Sub, Div}
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

    // backwards
    pub fn backward(&self) {
        
    }

    // Implementation of ReLU, Pow and tanh.
    pub fn relu(&self) -> Self {
        let mut relu_val = Value::new(self.data.max(0.0));
        relu_val.operation = String::from("ReLU");
        relu_val.previous = vec![self.clone()];
        relu_val.backward = Some(|val: &mut Value| {
            if val.data > 0.0 {
                val.previous[0].grad += val.grad;
            }
        });
        return relu_val;
    }

    pub fn pow(&self, n: f64) -> Self {
        let mut pow_val = Value::new(self.data.powf(n));
        pow_val.operation = String::from("pow()");
        pow_val.previous = vec![self.clone(), Value::new(n)];
        pow_val.backward = Some(|val: &mut Value|{
            let pow = val.previous[1].data;
            val.previous[0].grad += pow * val.grad * val.previous[0].data.powf(pow-1.0);
        });
        return pow_val;
    }

    pub fn tanh(&self) -> Self {
        let mut tanh_val = Value::new(self.data.tanh());
        tanh_val.operation = String::from("tanh()");
        tanh_val.previous = vec![self.clone()];
        tanh_val.backward = Some(|val: &mut Value| {
            let data_tanh = val.data.tanh();
            val.previous[0].grad += val.grad * (1.0 - data_tanh * data_tanh);
        });
        return tanh_val;
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
            val.previous[0].grad += val.grad * val.previous[1].data;
            val.previous[1].grad += val.grad * val.previous[0].data;
        });
        return mul_val;
    }
}

impl Div for Value {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let mut div_val = Value::new(self.data/other.data);
        div_val.operation = String::from("/");
        div_val.previous = vec![self, other];
        div_val.backward = Some(|val: &mut Value| {
            val.previous[0].grad += val.grad / val.previous[1].data;
            val.previous[1].grad -= val.grad * val.previous[0].data / val.previous[1].data.powi(2);
        });
        return div_val;
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