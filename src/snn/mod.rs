pub mod neuron;
pub mod lif_neuron;
pub mod layer;
pub mod spiking_n_n;

#[derive(Debug)]
pub struct SpikeEvent {
    ts: u64,            /* discrete time instant */
    spikes: Vec<u8>,    /* vector of spikes in that instant (a 1/0 for each input neuron)  */
}

impl SpikeEvent {
    pub fn new(ts: u64, spikes: Vec<u8>) -> Self {
        Self { ts, spikes }
    }
}