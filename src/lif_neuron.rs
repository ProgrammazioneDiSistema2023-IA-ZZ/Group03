/** LIF module */
use crate::neuron::Neuron;
use std::f64::consts::E;

/** Model of LIF Neuron (Leaky Integrate-and-Fire) */
#[derive(Debug)]
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
}

impl Neuron for LifNeuron {
    /*
        This function updates the membrane potential of the neuron when it receives at least one spike
    */
    fn get_v_mem(&mut self, t: u64, extra_sum: f64, intra_sum: f64) -> u8 {
        let weighted_sum = extra_sum + intra_sum;

        /* compute the neuron membrane potential with the LIF formula */
        let diff_time = (t - self.ts) as f64;
        let exponent = -(diff_time * self.dt / self.tau);
        self.v_mem = self.v_rest + (self.v_mem - self.v_rest) * E.powf(exponent) + weighted_sum;

        /* update ts at the last instant in which one spike (1) is received */
        self.ts = t;

        return if self.v_mem > self.v_th {
            self.v_mem = self.v_reset;  /* reset membrane potential */
            1
        } else {
            0
        };
    }

    fn init(&mut self) {
        self.v_mem = self.v_rest;
        self.ts = 0u64;
    }
}