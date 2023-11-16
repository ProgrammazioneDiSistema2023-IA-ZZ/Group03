/** Neuron module */

/** Trait for the implementation of any Neuron models */
pub trait Neuron: Send {
    /** The neuron function is invoked when incoming spikes from the previous layer.
        t: the time instant when input spikes are received
        extra_sum: the product of input spikes and incoming weights, always > 0
        intra_sum: the product of input spikes from the previous instant and the intra-layer weights
     */
    fn get_v_mem(&mut self, t: u64, extra_sum: f64, intra_sum: f64) -> u8;

    /** initialize all data structures of Neuron */
    fn init(&mut self);
}
