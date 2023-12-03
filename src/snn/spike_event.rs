#[derive(Debug)]
pub struct SpikeEvent {
    ts: u64,            /* discrete time instant */
    spikes: Vec<u8>,    /* vector of spikes in that instant (a 1/0 for each input neuron)  */
}

impl SpikeEvent {
    pub fn new(ts: u64, spikes: Vec<u8>) -> Self {
        Self { ts, spikes }
    }

    pub fn get_ts(&self) -> u64 {
        self.ts
    }

    pub fn get_spikes(&self) -> Vec<u8> {
        self.spikes.clone()
    }
}

