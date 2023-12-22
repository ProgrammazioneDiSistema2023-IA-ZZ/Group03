use crate::configuration::Configuration;

#[derive(Debug, Clone, PartialEq)]
pub struct Conf {
    components: Vec<String>,
    failure: Failure,
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
    position: u32,
    value: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StuckAt0 {
    position: u32,
    value: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TransientBitFlip {
    position: u32,
    bit_changed: bool,
}

impl Conf {
    pub fn new(components: Vec<String>, failure: Failure) -> Self {
        Self { components, failure }
    }
    pub fn get_components(&self) -> Vec<String> { self.components.clone() }
    pub fn get_failure(&self) -> Failure { self.failure.clone() }
}

impl StuckAt0 {
    pub fn new(position: u32) -> Self { Self { position, value: 0 } }
    pub fn get_position(&self) -> u32 { self.position }
    pub fn get_value(&self) -> u8 { self.value }
}

impl StuckAt1 {
    pub fn new(position: u32) -> Self { Self { position, value: 1 } }
    pub fn get_position(&self) -> u32 { self.position }
    pub fn get_value(&self) -> u8 { self.value }
}

impl TransientBitFlip {
    pub fn new(position: u32) -> Self { Self { position, bit_changed: false } }
    pub fn get_position(&self) -> u32 { self.position }
    pub fn get_bit_changed(&self) -> bool { self.bit_changed }
    pub fn set_bit_changed(&mut self, val: bool) { self.bit_changed = val }
}

impl Configuration for Conf {
    fn init(&mut self) {
        self.components = vec![];
        self.failure = Failure::None;
    }

    fn get_vec_components(&self) -> Vec<String> { self.components.clone() }
    fn get_len_vec_components(&self) -> usize { self.components.len() }
    fn get_failure(&self) -> Failure { self.failure.clone() }
}