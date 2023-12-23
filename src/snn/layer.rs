use std::sync::mpsc::{Receiver, Sender};
use crate::snn::neuron::Neuron;
use crate::snn::spike_event::SpikeEvent;
use crate::snn::configuration::Configuration;
use crate::failure::{Failure};

#[derive(Debug)]
pub struct Layer<N: Neuron + Clone + Send + 'static, R: Configuration + Clone + Send + 'static> {
    neurons: Vec<N>,
    weights: Vec<Vec<f64>>,
    intra_weights: Vec<Vec<f64>>,
    prev_spikes: Vec<u8>,
    configuration: R,
}

impl<N: Neuron + Clone + Send + 'static, R: Configuration + Clone + Send + 'static> Layer<N, R> {
    pub fn new(
        neurons: Vec<N>,
        weights: Vec<Vec<f64>>,
        intra_weights: Vec<Vec<f64>>,
        configuration: R,
    ) -> Self {
        let num_neurons = neurons.len();
        Self {
            neurons,
            weights,
            intra_weights,
            prev_spikes: vec![0; num_neurons],
            configuration,
        }
    }

    /* Getters  */
    pub fn get_number_neurons(&self) -> usize {
        self.neurons.len()
    }

    pub fn get_neurons(&self) -> Vec<N> { self.neurons.clone() }

    pub fn get_weights(&self) -> Vec<Vec<f64>> {
        self.weights.clone()
    }

    pub fn get_intra_weights(&self) -> Vec<Vec<f64>> {
        self.intra_weights.clone()
    }

    pub fn get_prev_spikes(&self) -> Vec<u8> { self.prev_spikes.clone() }

    pub fn get_configuration(&self) -> R { self.configuration.clone() }

    pub fn set_intra_weights(&mut self, val: Vec<Vec<f64>>) { self.intra_weights = val }

    fn generate_fault(&mut self) {
        let elements = self.configuration.get_vec_components();
        match self.configuration.get_failure() {
            Failure::StuckAt0(stuck_at_0) => {
                if elements.contains(&"v_th".to_string()) {
                    let neuron = self.neurons.get_mut(0).unwrap();
                    let mut vec_byte_original: Vec<u8> = neuron.get_v_th().to_ne_bytes().iter().cloned().collect();
                    modify_bits(&mut vec_byte_original, stuck_at_0.get_position() as u8 % 64u8, stuck_at_0.get_value());
                    neuron.set_v_th(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                }
                if elements.contains(&"intra_weights".to_string()) {
                    let mut matrix = self.intra_weights.clone();
                    let i: usize = ((stuck_at_0.get_position() / 64) / matrix.len() as u32) as usize;
                    let j: usize = ((stuck_at_0.get_position() / 64) % matrix.len() as u32) as usize;
                    let mut vec_byte = matrix[i][j].to_ne_bytes().iter().cloned().collect();
                    modify_bits(&mut vec_byte, stuck_at_0.get_position() as u8 % 64u8, stuck_at_0.get_value());
                    matrix[i][j] = f64::from_ne_bytes(vec_byte.as_slice().try_into().unwrap());
                    self.intra_weights = matrix;
                }
            }
            Failure::StuckAt1(stuck_at_1) => {
                if elements.contains(&"v_mem".to_string()) {
                    let neuron = self.neurons.get_mut(0).unwrap();
                    let mut vec_byte_original: Vec<u8> = neuron.get_v_mem().to_ne_bytes().iter().cloned().collect();
                    modify_bits(&mut vec_byte_original, stuck_at_1.get_position() as u8 % 64u8, stuck_at_1.get_value());
                    neuron.set_v_mem(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                }
            }
            Failure::TransientBitFlip(mut transient_bit_flip) => {
                if !transient_bit_flip.get_bit_changed() && elements.contains(&"v_th".to_string()) {
                    let neuron = self.neurons.get_mut(0).unwrap();
                    let mut vec_byte_original: Vec<u8> = neuron.get_v_th().to_ne_bytes().iter().cloned().collect();
                    let byte_original = vec_byte_original.get(transient_bit_flip.get_position() as usize / 8).cloned().unwrap_or(0u8);
                    let value_bit = (byte_original >> (transient_bit_flip.get_position() % 8)) & 1;
                    modify_bits(&mut vec_byte_original, transient_bit_flip.get_position() as u8 % 64u8, value_bit);
                    transient_bit_flip.set_bit_changed(true);
                    neuron.set_v_th(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                }
            }
            Failure::None => {}
        }
    }

    fn generate_spike(&mut self, input_spike_event: &SpikeEvent, instant: u64,
                      output_spikes: &mut Vec<u8>, at_least_one_spike: &mut bool) {
        /* Generate N-occurrences of faults according to the configuration */
        if self.configuration.get_len_vec_components() > 0 {
            for _ in 0..self.configuration.get_numbers_of_fault() {
                /* Forse meglio mettere il for e trasformarla in generate_faults() ??*/
                self.generate_fault();
            }
        }

        for (index, neuron) in self.neurons.iter_mut().enumerate() {
            let events = input_spike_event.get_spikes();

            /* compute extra weighted sum */
            let mut extra_weighted_sum = 0f64;
            let extra_weights_pairs = self.weights[index].iter().zip(events.iter());
            for (weight, spike) in extra_weights_pairs {
                if *spike != 0 {
                    extra_weighted_sum += *weight;
                }
            }

            /* compute intra weighted sum */
            let mut intra_weighted_sum = 0f64;
            let intra_weights_pairs = self.intra_weights[index].iter().zip(events.iter());
            for (i, (weight, spike)) in intra_weights_pairs.enumerate() {
                /* skip the reflexive link */
                if i != index && *spike != 0 {
                    intra_weighted_sum += *weight;
                }
            }

            let neuron_spike = neuron.calculate_v_mem(instant, extra_weighted_sum + intra_weighted_sum);
            output_spikes.push(neuron_spike);
            if neuron_spike == 1u8 {
                *at_least_one_spike = true;
            }
        }
    }

    pub fn process(&mut self, layer_input_rc: Receiver<SpikeEvent>, layer_output_tx: Sender<SpikeEvent>) {
        /* initialize data structures, so that the SNN can be reused */
        self.init();

        /* listen to SpikeEvent(s) coming from the previous layer and process them */
        while let Ok(input_spike_event) = layer_input_rc.recv() {
            /* time instant of the input spike */
            let instant = input_spike_event.get_ts();

            /*flag to manage not firing layer*/
            let mut at_least_one_spike = false;

            let mut output_spikes = Vec::<u8>::with_capacity(self.neurons.len());

            /*this function compute the v_mem for all the neurons considering the input spikes
            and if there is configuration field it provides to apply the fault */
            self.generate_spike(&input_spike_event, instant, &mut output_spikes, &mut at_least_one_spike);

            /* save output spikes for later */
            self.prev_spikes = output_spikes.clone();

            /* check if at least one neuron fired - if not, not send any spike */
            if !at_least_one_spike {
                continue;
            }

            /* at least one neuron fired -> send output spikes to the next layer */
            let output_spike_event = SpikeEvent::new(instant, output_spikes);

            layer_output_tx.send(output_spike_event)
                .expect(&format!("Unexpected error sending input spike event t={}", instant));
        }
    }

    pub fn init(&mut self) {
        self.prev_spikes.clear();    /* reset prev_spikes */
        self.neurons.iter_mut().for_each(|neuron| neuron.init());  /* reset neurons */
    }
}

pub fn modify_bits(vec_byte: &mut Vec<u8>, position: u8, val: u8) {
    let rest_position = 8 - (position % 8);
    for (pos, byte) in vec_byte.iter_mut().rev().enumerate() {
        if (position / 8) == pos as u8 {
            //check of the mask
            match 1_u8.checked_shl(rest_position as u32) {
                None => {}
                Some(value) => { *byte = (*byte & !value) | (val << rest_position); }
            }
        }
    }
}

impl<N: Neuron + Clone + Send + 'static, R: Configuration + Clone + Send + 'static> Clone for Layer<N, R> {
    fn clone(&self) -> Self {
        Self {
            neurons: self.neurons.clone(),
            weights: self.weights.clone(),
            intra_weights: self.intra_weights.clone(),
            prev_spikes: self.prev_spikes.clone(),
            configuration: self.configuration.clone(),
        }
    }
}