/** LIF module */
use crate::neuron::Neuron;
use std::f64::consts::E;

/** Model of LIF Neuron (Leaky Integrate-and-Fire) */
#[derive(Debug,Clone,PartialEq)]
pub struct LifNeuron {
    v_th: f64,      /* threshold potential */
    v_rest: f64,    /* resting potential */
    v_reset: f64,   /* reset potential */
    tau: f64,
    dt: f64,        /* time interval between two consecutive instants */
    v_mem: f64,     /* membrane potential */
    ts: u64,        /* last instant in which has been received at least one spike */
}

impl LifNeuron {
    pub fn new(v_th: f64, v_rest: f64, v_reset: f64, tau: f64, dt: f64) -> Self {
        Self {
            v_th,
            v_rest,
            v_reset,
            tau,
            dt,
            v_mem: v_rest,
            ts: 0u64,
        }
    }
    pub fn get_v_mem(&self)->f64{
        self.v_mem
    }

    pub fn get_ts(&self)->u64{
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

    fn set_v_mem(&mut self, intra: f64) {
        self.v_mem += intra;
    }
    /*
    This function updates the membrane potential of the neuron when it receives at least one spike
*/
    fn print_lif_neuron(&self) {
        println!("v_th : {}, v_rest : {}, v_reset : {}, tau : {}, dt : {}, v_mem: {}, ts: {}",
                 self.get_v_th(), self.get_v_rest(), self.get_v_reset(), self.get_tau(),
                 self.get_dt(), self.get_v_mem(), self.get_ts() );
    }
    fn calculate_v_mem(&mut self, t: u64, extra_sum: f64) -> u8 {
        let diff_time = (t - self.ts) as f64;
        let mut exponent=0.0;
        if diff_time != 0.0 && self.tau != 0.0 && self.dt != 0.0 {
            exponent = -(diff_time * self.dt / self.tau);
        }
        /** TODO!
                          self.dt
         */
        let mut esp :f64= 0.0;
        if (t as f64 - self.ts as f64) != 0.0 && self.tau != 0.0 {
            esp = (t as f64 - self.ts as f64)/self.tau ;
        }
        let base: f64 = 1.0 - 0.368;
        self.v_mem =  self.v_mem*( base.powf(esp) );
        if self.v_mem < self.v_rest {
            self.v_mem = self.v_rest;
        }
        self.v_mem = self.v_rest + (self.v_mem - self.v_rest) * E.powf(exponent) + extra_sum;
        self.ts = t;

        return if self.v_mem > self.v_th {
            self.v_mem = self.v_reset;
            1
        } else {
            return 0;
        };
    }

    fn init(&mut self) {
        self.v_mem = self.v_rest;
        self.ts = 0u64;
    }
}
