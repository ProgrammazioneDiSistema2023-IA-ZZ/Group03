use std::io::Write;
use std::sync::mpsc::{Receiver, Sender};
use crate::snn::neuron::Neuron;
use crate::snn::spike_event::SpikeEvent;
use crate::snn::configuration::Configuration;
use crate::failure::{Failure, Stuck_at_1, Transient_bit_flip};

#[derive(Debug)]
pub struct Layer<N: Neuron + Clone + Send + 'static, R: Configuration + Clone + Send + 'static> {
    neurons: Vec<N>,
    weights: Vec<Vec<f64>>,
    intra_weights: Vec<Vec<f64>>,
    prev_spikes: Vec<u8>,
    configuration: R,
}

impl<N: Neuron + Clone + Send + 'static, R: Configuration + Clone + Send + 'static> Layer<N,R> {
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

    fn modify_v_mem(&mut self, index: usize) {
        for index2 in 0..self.neurons.len(){
            if index2 != index-1 {
                let val = self.intra_weights.get(index-1).unwrap().get(index2).unwrap().clone();
                self.neurons[index2].decrement_v_mem(val);
            }
        }
    }
    fn continue_gen_spike(&mut self, input_spike_event: &SpikeEvent, instant:u64, output_spikes: &mut Vec<u8>,
                          at_least_one_spike: &mut bool, index_outside: &mut usize) {
        for (index, neuron) in self.neurons.iter_mut().skip(*index_outside).enumerate() {
            let mut extra_weighted_sum = 0f64;
            /* compute extra weighted sum */
            let events = input_spike_event.get_spikes();

            if self.configuration.get_len_vec_components() > 0 {
                let elementi = self.configuration.get_vec_components();
                match self.configuration.get_failure() {
                    Failure::StuckAt0(stuck_at_0) => {
                        if elementi.contains(&"v_th".to_string()) {
                            let mut vec_byte_original: Vec<u8> = neuron.get_v_th().to_ne_bytes().iter().cloned().collect();
                            neuron.modify_bits(&mut vec_byte_original,stuck_at_0.get_position(),stuck_at_0.get_valore());
                            neuron.set_v_th(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                        }
                    },
                    Failure::StuckAt1(stuck_at_1) => {
                        if elementi.contains(&"v_mem".to_string()) {
                            let mut vec_byte_original: Vec<u8> = neuron.get_v_th().to_ne_bytes().iter().cloned().collect();
                            neuron.modify_bits(&mut vec_byte_original,stuck_at_1.get_position(),stuck_at_1.get_valore());
                            neuron.set_v_mem(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                        }
                    }
                    Failure::TransientBitFlip(mut transient_bit_flip) => {
                        if !transient_bit_flip.get_bit_changed() {
                            if elementi.contains(&"v_th".to_string()) {
                                let mut vec_byte_original:Vec<u8> = neuron.get_v_th().to_ne_bytes().iter().cloned().collect();
                                let byte_original = vec_byte_original.get(transient_bit_flip.get_position() as usize / 8).cloned().unwrap_or(0u8);
                                let valor_bit = (byte_original >> (transient_bit_flip.get_position() % 8)) & 1;
                                neuron.modify_bits(&mut vec_byte_original,transient_bit_flip.get_position(),valor_bit);
                                transient_bit_flip.set_bit_changed(true);
                                neuron.set_v_th(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                            }
                        }
                    }
                    Failure::None => {}
                }
            }
            let extra_weights_pairs = self.weights[index + *index_outside].iter().zip(events.iter());

            for (weight, spike) in extra_weights_pairs {
                if *spike != 0 {
                    extra_weighted_sum += *weight;
                }
            }
            let neuron_spike = neuron.calculate_v_mem(instant, extra_weighted_sum);
            if self.configuration.get_len_vec_components() > 0 {
                let elementi = self.configuration.get_vec_components();
                match self.configuration.get_failure() {
                    Failure::StuckAt0(stuck_at_0) => {
                        if elementi.contains(&"v_th".to_string()) {
                            let mut vec_byte_original: Vec<u8> = neuron.get_v_th().to_ne_bytes().iter().cloned().collect();
                            neuron.modify_bits(&mut vec_byte_original,stuck_at_0.get_position(),stuck_at_0.get_valore());
                            neuron.set_v_th(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                        }
                    },
                    Failure::StuckAt1(stuck_at_1) => {
                        if elementi.contains(&"v_mem".to_string()) {
                            let mut vec_byte_original: Vec<u8> = neuron.get_v_th().to_ne_bytes().iter().cloned().collect();
                            neuron.modify_bits(&mut vec_byte_original,stuck_at_1.get_position(),stuck_at_1.get_valore());
                            neuron.set_v_mem(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                        }
                    }
                    Failure::TransientBitFlip(mut transient_bit_flip) => {
                        if !transient_bit_flip.get_bit_changed() {
                            if elementi.contains(&"v_th".to_string()) {
                                let mut vec_byte_original:Vec<u8> = neuron.get_v_th().to_ne_bytes().iter().cloned().collect();
                                let byte_original = vec_byte_original.get(transient_bit_flip.get_position() as usize / 8).cloned().unwrap_or(0u8);
                                let valor_bit = (byte_original >> (transient_bit_flip.get_position() % 8)) & 1;
                                neuron.modify_bits(&mut vec_byte_original,transient_bit_flip.get_position(),valor_bit);
                                transient_bit_flip.set_bit_changed(true);
                                neuron.set_v_th(f64::from_ne_bytes(vec_byte_original.as_slice().try_into().unwrap()));
                            }
                        }
                    }
                    Failure::None => {}
                }
            }
            output_spikes.push(neuron_spike);
            if neuron_spike == 1u8 {
                *at_least_one_spike = true;
                *index_outside = index+1 + *index_outside;
                break;
            }
        }
        *index_outside = self.neurons.len() ;
    }
    pub fn process(&mut self, layer_input_rc: Receiver<SpikeEvent>, layer_output_tx: Sender<SpikeEvent>) {
        /* initialize data structures, so that the SNN can be reused */
        self.init();
        /* listen to SpikeEvent(s) coming from the previous layer and process them */
        while let Ok(input_spike_event) = layer_input_rc.recv() {
            let instant = input_spike_event.get_ts();    /* time instant of the input spike */
            let mut output_spikes = Vec::<u8>::with_capacity(self.neurons.len());
            let mut at_least_one_spike = false;
            let mut index_outside:usize = 0;

            while index_outside < self.neurons.len() {
                self.continue_gen_spike(&input_spike_event, instant, output_spikes.by_ref(),
                                        &mut at_least_one_spike, &mut index_outside);
                if index_outside> 0 && index_outside < self.neurons.len() {
                    self.modify_v_mem(index_outside);
                }
            }
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

impl<N: Neuron + Clone + Send + 'static, R: Configuration + Clone + Send + 'static> Clone for Layer<N,R> {
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