/** LIF module */
use crate::neuron::Neuron;
use std::f64::consts::E;

/** Model of LIF Neuron (Leaky Integrate-and-Fire) */
#[derive(Debug, Clone, PartialEq)]
pub struct LifNeuron {
    v_th: f64,      /* Threshold potential */
    v_rest: f64,    /* Resting potential */
    v_reset: f64,   /* Reset potential */
    tau: f64,
    v_mem: f64,     /* Membrane potential */
    ts: u64,        /* Last instant in which has been received at least one spike */
    dt: f64,
}

impl LifNeuron {
    pub fn new(v_th: f64, v_rest: f64, v_reset: f64, tau: f64, dt: f64) -> Self {
        Self {
            v_th,
            v_rest,
            v_reset,
            tau,
            v_mem: v_rest,
            ts: 0u64,
            dt,
        }
    }
    pub fn get_v_mem(&self) -> f64 {
        self.v_mem
    }
    pub fn get_ts(&self) -> u64 {
        self.ts
    }
    pub fn get_v_th(&self) -> f64 {
        self.v_th
    }
    pub fn get_v_rest(&self) -> f64 {
        self.v_rest
    }
    pub fn get_v_reset(&self) -> f64 {
        self.v_reset
    }
    pub fn get_tau(&self) -> f64 {
        self.tau
    }
    pub fn get_dt(&self) -> f64 {
        self.dt
    }
}

impl Neuron for LifNeuron {
    fn get_v_th(&self) -> f64 {
        self.v_th
    }
    fn set_v_th(&mut self, new_val: f64) { self.v_th = new_val }
    fn print_lif_neuron(&self) {
        println!("v_th : {}, v_rest : {}, v_reset : {}, tau : {}, v_mem: {}, ts: {}",
                 self.get_v_th(), self.get_v_rest(), self.get_v_reset(), self.get_tau(),
                 self.get_v_mem(), self.get_ts());
    }
    fn calculate_v_mem(&mut self, t: u64, extra_intra_sum: f64) -> u8 {
        let diff_time;
        if t > self.ts {
            diff_time = (t - self.ts) as f64;
        }
        else {
            diff_time = (self.ts - t) as f64;
        }
        let mut exponent = 0.0;
        if diff_time != 0.0 && self.tau != 0.0 {
            exponent = -(diff_time * self.dt / self.tau);
        }
        if self.v_mem < self.v_rest {
            self.v_mem = self.v_rest;
        }
        self.v_mem = self.v_rest + (self.v_mem - self.v_rest) * E.powf(exponent) + extra_intra_sum;
        self.ts = t;
        if self.v_mem > self.v_th {
            self.v_mem = self.v_reset;
            1
        } else {
            0
        }
    }
    fn init(&mut self) {
        self.v_mem = self.v_rest;
        self.ts = 0u64;
    }
    fn get_tau(&self) -> f64 {
        self.tau
    }
    fn get_v_reset(&self) -> f64 {
        self.v_reset
    }
    fn get_v_rest(&self) -> f64 {
        self.v_rest
    }
    fn get_ts(&self) -> u64 {
        self.ts
    }
    fn get_v_mem(&self) -> f64 {
        self.v_mem
    }
    fn set_v_mem(&mut self, val: f64) {
        self.v_mem = val;
    }
    fn set_tau(&mut self, val: f64) {
        self.tau = val;
    }
    fn set_v_reset(&mut self, val: f64) {
        self.v_reset = val;
    }
    fn set_v_rest(&mut self, val: f64) {
        self.v_rest = val;
    }
    /* This function updates the membrane potential of the neuron when it receives at least one spike */
    fn set_ts(&mut self, val: u64) {
        self.ts = val;
    }
    fn get_dt(&self) -> f64 {
        self.dt
    }
    fn set_dt(&mut self, val: f64) {
        self.dt = val;
    }
}