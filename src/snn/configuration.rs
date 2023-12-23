use crate::failure::Failure;
/** Configuration module */
/** Trait for the implementation of any failure configuration */
pub trait Configuration: Send {
    fn init(&mut self);
    fn get_vec_components(&self) -> Vec<String> ;
    fn get_len_vec_components(&self) -> usize;
    fn get_failure(&self) -> Failure;
    fn get_numbers_of_fault(&self) -> usize;
}
