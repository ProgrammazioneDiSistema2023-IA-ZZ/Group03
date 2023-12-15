use crate::configuration::Configuration;
#[derive(Debug,Clone,PartialEq)]
pub struct Conf {
    components:Vec<String>,
    failure: Failure,
}
#[derive(Debug,Clone,PartialEq)]
pub enum Failure{
    StuckAt0(Stuck_at_0),
    StuckAt1(Stuck_at_1),
    TransientBitFlip(Transient_bit_flip),
    None,
}
#[derive(Debug,Clone,PartialEq)]
pub struct Stuck_at_1 {
    position: u8,
    valore: u8,
}
#[derive(Debug,Clone,PartialEq)]
pub struct Stuck_at_0 {
    position: u8,
    valore: u8,
}
#[derive(Debug,Clone,PartialEq)]
pub  struct Transient_bit_flip {
    position: u8,
    bit_gia_cambiato : bool,
}
impl Conf {
    pub fn new(components: Vec<String>, failure: Failure) -> Self {
        Self { components, failure }
    }
    pub fn get_components(&self) ->Vec<String> { self.components.clone() }
    pub fn get_failure(&self) -> Failure { self.failure.clone() }
}

impl Stuck_at_0 {
    pub fn new(position:u8) -> Self { Self {position,valore:0} }
    pub fn get_position(&self) -> u8 { self.position }
    pub fn get_valore(&self) -> u8 { self.valore }
}

impl Stuck_at_1 {
    pub fn new(position:u8) -> Self { Self {position,valore:1} }
    pub fn get_position(&self) -> u8 { self.position }
    pub fn get_valore(&self) -> u8 { self.valore }
}
impl Transient_bit_flip {
    pub fn new(position:u8) -> Self { Self {position, bit_gia_cambiato: false } }
    pub fn get_position(&self) -> u8 { self.position }
    pub fn get_bit_changed(&self) -> bool { self.bit_gia_cambiato }
    pub fn set_bit_changed(&mut self,val:bool){ self.bit_gia_cambiato = val }
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