use crate::failure::{Components, Failure};
/** Configuration module */
/** Trait for the implementation of any failure configuration */
pub trait Configuration: Send {
    fn init(&mut self);
    fn get_vec_components(&self) -> Vec<Components> ;
    fn get_len_vec_components(&self) -> usize;
    fn get_failure(&self) -> Failure;
    fn get_index_neuron(&self) -> usize;
    fn set_done(&self, val : bool);
    fn get_done(&self) -> bool;
}
