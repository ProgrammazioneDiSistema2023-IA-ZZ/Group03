use std::fs::{File};
use std::io::{BufRead, BufReader, Write};
use rand::{random, Rng, thread_rng};
use spiking_neural_network::failure::*;
use spiking_neural_network::lif_neuron::LifNeuron;
use spiking_neural_network::snn::builder::SnnBuilder;

//Accuracy: ['71.0%']
// let configuration = Conf::new(vec![Components::IntraWeights], Failure::StuckAt0(StuckAt0::new(144021)), 0);

//Accuracy: ['71.0%','69.0%','70.0%']
// let configuration = Conf::new(vec![], Failure::None, 0);

//Accuracy: ['11.0%'] vth troppo alta => la v_mem non supera mai vth => spikes=0
// let configuration = Conf::new(vec![Components::VTh], Failure::StuckAt1(StuckAt1::new(3)), 1);


const N_NEURONS: usize = 400;
const N_INPUTS: usize = 784;
const N_INSTANTS: usize = 3500;

fn get_current_dir()->String{
    let current_dir = std::env::current_dir().expect("Failed to get current directory");

    // Convert the PathBuf to a string
    let current_dir_str = current_dir.to_string_lossy();

    // Replace backslashes with forward slashes
    let path = current_dir_str.replace("\\", "/");

    path 
}

fn main(){
    let path = get_current_dir();

    let input_spikes: Vec<Vec<u8>> = read_input_spikes(&path);

    let neurons: Vec<LifNeuron> = build_neurons(&path);
    let extra_weights: Vec<Vec<f64>> = read_extra_weights(&path);
    let intra_weights: Vec<Vec<f64>> = build_intra_weights();

    let configuration = Conf::new(vec![], Failure::None, 0);

    let mut rng1 = thread_rng();
    let random_bit = rng1.gen();

    let mut rng2 = thread_rng();
    let random_index = rng2.gen_range(0..N_NEURONS);

    /* NEW CONFIGURATIONS */
    let configuration = Conf::new(vec![Components::VTh], Failure::StuckAt0(StuckAt0::new(random_bit)), random_index);
    // let configuration = Conf::new(vec![Components::VMem], Failure::TransientBitFlip(TransientBitFlip::new(37)), 211);
    // let configuration = Conf::new(vec![Components::Tau], Failure::StuckAt1(StuckAt1::new(21)), 176);
    // let configuration = Conf::new(vec![Components::VReset], Failure::StuckAt0(StuckAt0::new(56)), 14);
    // let configuration = Conf::new(vec![Components::VRest], Failure::TransientBitFlip(TransientBitFlip::new(40)), 72);
    // let configuration = Conf::new(vec![Components::Dt], Failure::StuckAt1(StuckAt1::new(10)), 345);
    // let configuration = Conf::new(vec![Components::IntraWeights], Failure::StuckAt0(StuckAt0::new(1234)), 0);
    // let configuration = Conf::new(vec![Components::Weights], Failure::StuckAt0(StuckAt0::new(2100)), 0);

    // println!("{:?}",configure_v_th.get_vec_components()[0]);

    let mut snn = SnnBuilder::new(N_INPUTS)
        .add_layer(neurons, extra_weights, intra_weights,configuration)
        .build();


    let output_spikes = snn.process(&input_spikes);

    write_to_output_file(output_spikes,&path);


}

/* * USEFUL FUNCTIONS * */

/**
This function builds the neurons of the network.
 */
fn build_neurons(path:&String) -> Vec<LifNeuron> {
    let thresholds: Vec<f64> = read_thresholds(path);

    let v_rest: f64 = -65.0;
    let v_reset: f64 = -60.0;
    let tau: f64 = 100.0;
    let dt: f64 = 0.1; 

    let mut neurons: Vec<LifNeuron> = Vec::with_capacity(N_NEURONS);

    for i in 0..N_NEURONS {
        let neuron = LifNeuron::new(thresholds[i], v_rest, v_reset, tau, dt);
        neurons.push(neuron);
    }

    neurons
}

/**
This function builds a 2D Vec of intra weights
 */
fn build_intra_weights() -> Vec<Vec<f64>> {
    let value: f64 = -15.0;
    //let value: f64 = 0.0;

    let mut intra_weights: Vec<Vec<f64>> = vec![vec![0f64; N_NEURONS]; N_NEURONS];


    for i in 0..N_NEURONS {
        for j in 0..N_NEURONS {
            if i == j {
                intra_weights[i][j] = 0.0;
            } else {
                intra_weights[i][j] = value;
            }
        }
    }

    intra_weights
}

/**
This function reads the input spikes from the input file and returns a 2D Vec of u8.
 */
fn read_input_spikes(path:&String) -> Vec<Vec<u8>> {

    let path_input = format!("{path}/inputSpikes.txt");

    let input = File::open(path_input).expect("Something went wrong opening the file inputSpikes.txt!");
    let buffered = BufReader::new(input);

    let mut input_spikes: Vec<Vec<u8>> = vec![vec![0; N_INSTANTS]; N_INPUTS];

    let mut i = 0;

    for line in buffered.lines() {
        let mut j = 0;
        let chars = convert_line_into_u8(line.unwrap());
        chars.into_iter().for_each(| ch | {
            input_spikes[j][i] = ch;
            j += 1;
        });
        i += 1;
    }

    input_spikes
}


/**
This function reads the weights file and returns a 2D Vec of weights
 */
fn read_extra_weights(path:&String) -> Vec<Vec<f64>> {
    let path_weights_file = format!("{path}/networkParameters/weightsOut.txt");

    let input = File::open(path_weights_file).expect("Something went wrong opening the file weightsOut.txt!");
    let buffered = BufReader::new(input);

    let mut extra_weights: Vec<Vec<f64>> = vec![vec![0f64; N_INPUTS]; N_NEURONS];

    let mut i = 0;

    for line in buffered.lines() {
        let split: Vec<String> = line.unwrap().as_str().split(" ").map(|el| el.to_string()).collect::<Vec<String>>();

        for j in 0..N_INPUTS {
            extra_weights[i][j] = split[j].parse::<f64>().expect("Cannot parse string into f64!");
        }

        i += 1;
    }

    extra_weights
}

/**
This function reads the threshold file and returns a Vec of thresholds
 */
fn read_thresholds(path:&String) -> Vec<f64> {

    let path_threshold_file = format!("{path}/networkParameters/thresholdsOut.txt");

    let input = File::open(path_threshold_file).expect("Something went wrong opening the file thresholdsOut.txt!");
    let buffered = BufReader::new(input);

    let mut thresholds: Vec<f64> = vec![0f64; N_NEURONS];

    let mut i = 0;

    for line in buffered.lines() {

        thresholds[i] = line.unwrap().parse::<f64>().expect("Cannot parse String into f64!");

        i += 1;
    }

    thresholds
}

/**
This function writes the output spikes to the output file.
 */
fn write_to_output_file(output_spikes: Vec<Vec<u8>>,path:&String) -> () {
    let path_output = format!("{path}/outputCounters.txt");

    let mut output_file = File::create(path_output).expect("Something went wrong opening the file outputCounters.txt!");

    let mut neurons_sum: Vec<u32> = vec![0; N_NEURONS];

    for i in 0..N_NEURONS {
        for j in 0..N_INSTANTS {
            neurons_sum[i] += output_spikes[i][j] as u32;
        }
    }

    for i in 0..N_NEURONS {
        output_file.write_all(format!("{}\n", neurons_sum[i]).as_bytes()).expect("Something went wrong writing into the file outputCounters.txt!");
    }

}


/**
This function converts a line of the input file into a Vec of u8.
 */
fn convert_line_into_u8(line: String) -> Vec<u8> {
    line.chars()
        .map(|ch| (ch.to_digit(10).unwrap()) as u8)
        .collect::<Vec<u8>>()
}



