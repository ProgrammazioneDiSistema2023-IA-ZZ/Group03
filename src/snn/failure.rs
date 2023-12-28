use crate::configuration::Configuration;

#[derive(Debug, Clone, PartialEq)]
pub struct Conf {
    components: Vec<Components>,
    failure: Failure,
    index_neuron: usize,
}

/*
#[derive(Debug, Clone, PartialEq)]
pub struct Conf {
    info: Vec<Components, usize>,
    failure: Failure,
}
*/

#[derive(Debug, Clone, PartialEq)]
pub enum Components {
    /* List of possible fault components of LifNeuron */
    VTh,
    VRest,
    VReset,
    Tau,
    VMem,
    Ts,
    Dt,

    /* List of possible fault components of Layers*/
    Weights,
    IntraWeights,
    PrevSpikes,

    None,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Failure {
    StuckAt0(StuckAt0),
    StuckAt1(StuckAt1),
    TransientBitFlip(TransientBitFlip),
    None,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StuckAt1 {
    position: usize,
    value: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StuckAt0 {
    position: usize,
    value: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransientBitFlip {
    position: usize,
    bit_changed: bool,
}


impl StuckAt0 {
    pub fn new(position: usize) -> Self { Self { position, value: 0 } }
    pub fn get_position(&self) -> usize { self.position }
    pub fn get_value(&self) -> u8 { self.value }
}

impl StuckAt1 {
    pub fn new(position: usize) -> Self { Self { position, value: 1 } }
    pub fn get_position(&self) -> usize { self.position }
    pub fn get_value(&self) -> u8 { self.value }
}

impl TransientBitFlip {
    pub fn new(position: usize) -> Self { Self { position, bit_changed: false } }
    pub fn get_position(&self) -> usize { self.position }
    pub fn get_bit_changed(&self) -> bool { self.bit_changed }
    pub fn set_bit_changed(&mut self, val: bool) { self.bit_changed = val }
}

impl Conf {
    pub fn new(components: Vec<Components>, failure: Failure, index_neuron: usize) -> Self {
        Self { components, failure, index_neuron }
    }
}

impl Configuration for Conf {
    fn init(&mut self) {
        self.components = vec![];
        self.failure = Failure::None;
    }
    fn get_vec_components(&self) -> Vec<Components> { self.components.clone() }
    fn get_len_vec_components(&self) -> usize { self.components.len() }
    fn get_failure(&self) -> Failure { self.failure.clone() }
    fn get_index_neuron(&self) -> usize { self.index_neuron }
}

impl Failure {
    pub fn get_position(&self) -> Option<usize> {
        match self {
            Failure::StuckAt0(s) => { Some(s.get_position()) }
            Failure::StuckAt1(s) => { Some(s.get_position()) }
            Failure::TransientBitFlip(t) => { Some(t.get_position()) }
            _ => { None }
        }
    }
}