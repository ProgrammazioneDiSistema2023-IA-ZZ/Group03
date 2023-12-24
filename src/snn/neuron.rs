/** Neuron module */

/** Trait for the implementation of any Neuron models */
pub trait Neuron: Send {
    /** The neuron function is invoked when incoming spikes from the previous layer.
        t: the time instant when input spikes are received
        extra_sum: the product of input spikes and incoming weights, always > 0
        intra_sum: the product of input spikes from the previous instant and the intra-layer weights
     */
    fn decrement_v_mem(&mut self, intra: f64);
    fn get_v_th(&self) -> f64;
    fn set_v_th(&mut self, new_val:f64);
    fn print_lif_neuron(&self) ;
    fn calculate_v_mem(&mut self, t: u64, extra_sum: f64) -> u8;

    /** initialize all data structures of Neuron */
    fn init(&mut self);
    fn get_tau(&self) -> f64;
    fn get_v_reset(&self) -> f64;
    fn get_v_rest(&self) -> f64;
    fn get_ts(&self)->u64;
    fn get_v_mem(&self)->f64;
    fn set_v_mem(&mut self, val: f64);
    fn set_tau(&mut self, val: f64);
    fn set_v_reset(&mut self, val: f64);
    fn set_v_rest(&mut self, val: f64);
    fn set_ts(&mut self, val: u64);
    fn get_dt(&self)->f64;
    fn set_dt(&mut self,val:f64);
}
