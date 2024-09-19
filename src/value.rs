use core::f64;
use std::{
    cell::RefCell, collections::HashSet, fmt::Debug, hash::Hash, iter::Sum, ops::{Add, Deref, Div, Mul, Sub}, rc::Rc
};
use uuid::Uuid;



/* Structure to hold data , gradient and required info for backpropagation.
 * Members:
 * id(Uuid):    Unique id for the node.
 * data(float): The value for the node.
 * grad(float): The gradient for the node.
 * operation(string): The opeartion performed to get to this node.
 *                    Currently supported operations are - arithmetic operations(+, -, *, /),
 *                    power, and activation functions ( ReLu and tanh)
 * backwards(optional, function): backward function pointer to calculate gardients
 *                                through chain rule.
 * previous(vector of Val): holds reference to previous nodes.
 *                          As we need to hold reference to previous nodes, we use struct Val
 *                          which is just a reference counted mutable reference to Value.
*/
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

/*===================================================*/
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

impl<T: Into<f64>> From<T> for Val {
    fn from(t: T) -> Val {
        Val::new(Value::new(t.into()))
    }
}

impl Debug for Val {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Value")
        .field("id", &self.borrow().id)
        .field("data", &self.borrow().data)
        .field("grad", &self.borrow().grad)
        .field("operation", &self.borrow().operation)
        .field("previous", &self.borrow().previous)
        .finish()
    }
}
/*===================================================*/

// Implementation of power, ReLU and tanh.
// Implementation of back propagation backward pass.
impl Val {
    fn new(val: Value) -> Val {
        Val(Rc::new(RefCell::new(val)))
    }

    // backwards
    pub fn backward(&self) {
        let mut topo = self.build_topology();
        topo.reverse();
        self.borrow_mut().grad = 1.0;
        for node in topo {
            if let Some(back) = node.borrow().backward {
                back(&node.borrow());
            }
        }
    }

    // Build topology
    fn build_topology(&self) -> Vec<Val> {
        let mut topo: Vec<Val> = vec![];
        let mut visited: HashSet<Val> = HashSet::new();
        self._build(&mut visited, &mut topo);
        return topo;
    }

    fn _build(&self, visited: &mut HashSet<Val>, topo: &mut Vec<Val>) {
        if !visited.contains(self) {
            visited.insert(self.clone());
            for child in self.borrow().previous.iter() {
                child._build(visited, topo);
            }
            topo.push(self.clone());
        }
    }

    // Implementation of ReLU, Pow and tanh.
    pub fn relu(&self) -> Self {
        let mut relu_val = Value::new(self.borrow().data.max(0.0));
        relu_val.operation = String::from("ReLU");
        relu_val.previous = vec![self.clone()];
        relu_val.backward = Some(|val: &Value| {
            if val.data > 0.0 {
                val.previous[0].borrow_mut().grad += val.grad;
            }
        });
        return Val::new(relu_val);
    }

    pub fn pow(&self, n: f64) -> Self {
        let mut pow_val = Value::new(self.borrow().data.powf(n));
        pow_val.operation = String::from("pow()");
        pow_val.previous = vec![self.clone(), Val::new(Value::new(n))];
        pow_val.backward = Some(|val: &Value|{
            let pow = val.previous[1].borrow().data;
            let base = val.previous[0].borrow().data;
            val.previous[0].borrow_mut().grad += pow * val.grad * base.powf(pow-1.0);
        });
        return Val::new(pow_val);
    }

    pub fn tanh(&self) -> Self {
        let mut tanh_val = Value::new(self.borrow().data.tanh());
        tanh_val.operation = String::from("tanh()");
        tanh_val.previous = vec![self.clone()];
        tanh_val.backward = Some(|val: &Value| {
            let data_tanh = val.data.tanh();
            val.previous[0].borrow_mut().grad += val.grad * (1.0 - data_tanh * data_tanh);
        });
        return Val::new(tanh_val);
    }
 }

/*=======================================================*/

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
        return Val::new(add_val);
    }
}

impl Sub for Val {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut sub_val = Value::new(self.borrow().data - other.borrow().data);
        sub_val.operation = String::from("-");
        sub_val.previous = vec![self, other];
        sub_val.backward = Some(|val: &Value| {
            val.previous[0].borrow_mut().grad += val.grad;
            val.previous[1].borrow_mut().grad -= val.grad;
        });
        return Val::new(sub_val);
    }
}

impl Mul for Val {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let mut mul_val = Value::new(self.borrow().data * other.borrow().data);
        mul_val.operation = String::from("*");
        mul_val.previous = vec![self, other];
        mul_val.backward = Some(|val: &Value| {
            val.previous[0].borrow_mut().grad += val.grad * val.previous[1].borrow().data;
            val.previous[1].borrow_mut().grad += val.grad * val.previous[0].borrow().data;
        });
        return Val::new(mul_val);
    }
}

impl Div for Val {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let mut div_val = Value::new(self.borrow().data/other.borrow().data);
        div_val.operation = String::from("/");
        div_val.previous = vec![self, other];
        div_val.backward = Some(|val: &Value| {
            val.previous[0].borrow_mut().grad += val.grad / val.previous[1].borrow().data;
            val.previous[1].borrow_mut().grad -= val.grad * val.previous[0].borrow().data / val.previous[1].borrow().data.powi(2);
        });
        return Val::new(div_val);
    }
}

impl Sum for Val {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut new_val = Value::new(0.0);

        let sum = iter
        .map(|val| {
            new_val.previous.push(val.clone());
            val.borrow().data
        }).sum();

        new_val.data = sum;
        new_val.operation = String::from("+");
        new_val.backward = Some(|val: &Value| {
            for v in val.previous.iter() {
                v.borrow_mut().grad += val.grad;
            }
        });

        Val::new(new_val)
    }
}