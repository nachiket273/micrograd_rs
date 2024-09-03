use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    fmt::Debug,
    hash::Hash,
    ops::{Add, Deref},
    rc::Rc
};
use uuid::Uuid;

// Basic Implementation of struct Value
#[derive(Debug)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
    pub operation: String,
    pub backward: Option<fn(val: &Value)>,
    pub previous: Vec<Val>,
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

#[derive(Clone)]
pub struct Val(Rc<RefCell<Value>>);

impl Deref for Val {
    type Target = Rc<RefCell<Value>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialEq for Val {
    fn eq(&self, other: &Self) -> bool {
        self.borrow().id == other.borrow().id
    }
}

impl Eq for Val{}

impl Hash for Val {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.borrow().id.hash(state)
    }
}

impl Debug for Val {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value")
        .field("data", &self.borrow().data)
        .field("grad", &self.borrow().grad)
        .field("operation", &self.borrow().operation)
        .finish()
    }
}


impl Val {
    fn new(value: Value) -> Val {
        Val(Rc::new(RefCell::new(value)))
    }
}

impl<T: Into<f64>> From<T> for Val {
    fn from(t: T) -> Val {
        Val::new(Value::new(t.into()))
    }
}

// Implementation of Add, Subtract, Multiply and Divide
impl Add for Val {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut add_val = Value::new(self.borrow().data + other.borrow().data);
        add_val.operation = String::from("+");
        add_val.previous = vec![self, other];
        add_val.backward = Some(|val: &Value| {
            val.previous[0].borrow_mut().grad += val.grad;
            val.previous[1].borrow_mut().grad += val.grad;
        });
        Val::new(add_val)
    }
}

fn main() {
    let val = Val::from(10.0);
    let val2 = Val::clone(&val);
    println!("Current struct: {:#?}", val);
}  