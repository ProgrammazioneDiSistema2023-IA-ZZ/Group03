mod demo;

use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use rand::{Rng, thread_rng};
use spiking_neural_network::failure::*;
use spiking_neural_network::lif_neuron::LifNeuron;
use spiking_neural_network::snn::builder::SnnBuilder;
use spiking_neural_network::snn::configuration::Configuration;
use std::process::{Command};
use std::thread;
use std::thread::JoinHandle;
use std::time::Instant;
use bit::BitIndex;
use zip::read::ZipArchive;
use crate::demo::demo;

const CYCLES: usize = 51;
const N_NEURONS: usize = 400;
const N_INPUTS: usize = 784;
const N_INSTANTS: usize = 3500;

fn main() {

    //demo();

    let path = get_current_dir();
    let result = read_zip(&path);
    match result {
        Ok(_zip) => { start_snn(); }
        Err(e) => { println!("{:?}", e); }
    }
}

fn start_snn() {
    let path = get_current_dir();
    let start = Instant::now();
    let mut threads = Vec::<JoinHandle<()>>::new();

    /* list of components to simulate */
    let vec_comp = vec![
        Components::Ts, Components::Dt, Components::Weights,
        Components::IntraWeights, Components::PrevSpikes,
        Components::VTh, Components::VMem, Components::VReset,
        Components::VRest, Components::Tau];

    /* repeats the simulation with 2 parallel threads on the same component but with different types of failures */
    for elem in vec_comp.clone() {
        /* compute random bit and random index to set fault in precise way */
        let mut rng1 = thread_rng();
        let random_bit = rng1.gen_range(0..12);
        let mut rng2 = thread_rng();
        let random_index = rng2.gen_range(0..N_NEURONS);

        for f in 0..3 {
            let elem_c = elem.clone();
            let path_clone = path.clone();

            /* build parameters of the network */
            let input_spikes: Vec<Vec<Vec<u8>>> = read_multiple_input_spikes(&path_clone);
            let neurons: Vec<LifNeuron> = build_neurons(&path_clone);
            let extra_weights: Vec<Vec<f64>> = read_extra_weights(&path_clone);
            let intra_weights: Vec<Vec<f64>> = build_intra_weights();

            let vec_type_fail = vec![
                Failure::StuckAt1(StuckAt1::new(random_bit)),
                Failure::StuckAt0(StuckAt0::new(random_bit)),
                Failure::TransientBitFlip(TransientBitFlip::new(random_bit)),
            ];

            /* checks if the simulation makes sense or if the selected bit
                is already at 0/1, in which case the thread will not spawned */
            let position = vec_type_fail[f].get_position().unwrap();
            let val = get_val(elem_c.clone(), neurons.clone(), random_index, intra_weights.clone(), extra_weights.clone(), position);
            let bit = val.bit(position);
            match vec_type_fail[f] {
                Failure::StuckAt0(_) if bit == false => {
                    println!("Useless simulation StuckAt0");
                    continue;
                }
                Failure::StuckAt1(_) if bit == true => {
                    println!("Useless simulation StuckAt1");
                    continue;
                }
                _ => {}
            }

            let thread = thread::spawn(move || {
                /* creates a Conf instance that will be inserted into the layer to report the fault  */
                let configuration = Conf::new(vec![elem_c.clone()], vec_type_fail[f].clone(), random_index);

                let file_name = get_file_name(&configuration);
                let path_output = format!("{path_clone}/simulation/configurations/{file_name}");
                let mut output_file = File::create(path_output).expect("Something went wrong opening the file outputCounters.txt!");

                /* run simulation over snn with fault configuration */
                for i in 0..CYCLES {
                    let mut snn = SnnBuilder::new(N_INPUTS)
                        .add_layer(neurons.clone(), extra_weights.clone(), intra_weights.clone(), configuration.clone())
                        .build();

                    /* print debug */
                    print!("Iteration {i}/50 - Component {:?} - ", elem_c);
                    match configuration.get_failure() {
                        Failure::StuckAt0(_) => { println!("StuckAt0"); }
                        Failure::StuckAt1(_) => { println!("StuckAt1"); }
                        Failure::TransientBitFlip(_) => { println!("TransientBitFlip"); }
                        _ => {}
                    }

                    let output_spikes = snn.process(&input_spikes[i]);
                    let mut neurons_sum = vec![0u32; 400];
                    for k in 0..N_NEURONS {
                        for j in 0..N_INSTANTS {
                            neurons_sum[k] += output_spikes[k][j] as u32;
                        }
                    }

                    for n in 0..N_NEURONS {
                        output_file.write_all(format!("{}\n", neurons_sum[n]).as_bytes()).expect("Something went wrong writing into the file outputCounters.txt!");
                    }
                }
            });

            /* push the new thread into pool of threads */
            threads.push(thread);
        }
    }

    /* waiting for threads to terminate */
    for thread in threads {
        thread.join().unwrap();
    }

    /* print duration */
    let time = start.elapsed();
    let min = time.as_secs() as f64 / 60.0;
    println!("Duration simulations: {:.2} min", min);

    /* when all simulations are finished run the python script to print logs file */
    let string = format!("{path}/simulation/runSimulation.py");
    let path_py = OsStr::new(&string);
    Command::new("python").arg(path_py).output()
        .expect("Error during execution of the command");
}

fn get_val(e: Components, neurons: Vec<LifNeuron>, index: usize, intra_weights: Vec<Vec<f64>>, extra_weights: Vec<Vec<f64>>, position: usize) -> usize {
    return match e {
        Components::VTh => {
            neurons.get(index).unwrap().get_v_th().to_bits()
        }
        Components::VRest => {
            neurons.get(index).unwrap().get_v_rest().to_bits()
        }
        Components::VReset => {
            neurons.get(index).unwrap().get_v_reset().to_bits()
        }
        Components::Tau => {
            neurons.get(index).unwrap().get_tau().to_bits()
        }
        Components::Dt => {
            neurons.get(index).unwrap().get_dt().to_bits()
        }
        Components::IntraWeights => {
            let matrix = intra_weights.clone();
            let i = (position / 64) / matrix.len();
            let j = (position / 64) % matrix.len();
            matrix[i][j].to_bits()
        }
        Components::Weights => {
            let matrix = extra_weights.clone();
            let i = (position / 64) / matrix.len();
            let j = (position / 64) % matrix.len();
            matrix[i][j].to_bits()
        }
        _ => { 0 }
    } as usize;
}

fn get_file_name(conf: &Conf) -> String {
    let components = conf.get_vec_components();
    let comp = components.get(0).unwrap_or(&Components::None).clone();

    let comp_string = match comp {
        Components::VTh => { "VTh" }
        Components::VRest => { "VRest" }
        Components::VReset => { "VReset" }
        Components::Tau => { "Tau" }
        Components::VMem => { "VMem" }
        Components::Ts => { "Ts" }
        Components::Dt => { "Dt" }
        Components::Weights => { "Weights" }
        Components::IntraWeights => { "IntraWeights" }
        Components::PrevSpikes => { "PrevSpikes" }
        _ => { "NoFault" }
    };

    let failure = conf.get_failure();

    let failure_string =
        match failure {
            Failure::StuckAt0(s) => { format!("StuckAt0_{}", s.get_position()) }
            Failure::StuckAt1(s) => { format!("StuckAt1_{}", s.get_position()) }
            Failure::TransientBitFlip(t) => { format!("Transient_{}", t.get_position()) }
            Failure::None => { "None".to_string() }
        };
    let neuron_index = conf.get_index_neuron();

    format!("{comp_string}_{failure_string}_{neuron_index}.txt")
}

fn read_multiple_input_spikes(path: &String) -> Vec<Vec<Vec<u8>>> {
    let path_input = format!("{path}/simulation/inputSpikes.txt");
    let input = File::open(path_input).expect("Something went wrong opening the file inputSpikes.txt!");
    let buffered = BufReader::new(input);

    let mut vec_input_spikes: Vec<Vec<Vec<u8>>> = vec![vec![vec![0; N_INSTANTS]; N_INPUTS]; CYCLES];

    let mut i = 0;
    let mut count = 0;
    let mut input_spikes: Vec<Vec<u8>> = vec![vec![0; N_INSTANTS]; N_INPUTS];

    for line in buffered.lines() {
        let c = '\n';
        let l = line.unwrap().replace(c, "");
        let mut j = 0;
        let chars = convert_line_into_u8(l);
        chars.into_iter().for_each(|ch| {
            input_spikes[j][i] = ch;
            j += 1;
        });

        i += 1;

        if i % N_INSTANTS == 0 && i > 0 {
            i = 0;
            vec_input_spikes[count] = input_spikes.clone();
            count += 1;
            if count == CYCLES {
                break;
            }
        }
    }
    vec_input_spikes
}

/* * USEFUL FUNCTIONS * */

/**
This function builds the neurons of the network.
 */
fn build_neurons(path: &String) -> Vec<LifNeuron> {
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

/*
This function reads the weights file and returns a 2D Vec of weights
 */
fn read_extra_weights(path: &String) -> Vec<Vec<f64>> {
    let path_weights_file = format!("{path}/simulation/networkParameters/weightsOut.txt");

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

fn read_thresholds(path: &String) -> Vec<f64> {
    let path_threshold_file = format!("{path}/simulation/networkParameters/thresholdsOut.txt");

    let input = File::open(path_threshold_file).expect("Something went wrong opening the file thresholdsOut.txt!");
    let buffered = BufReader::new(input);

    let mut thresholds: Vec<f64> = vec![0f64; N_NEURONS];

    for (i, line) in buffered.lines().enumerate() {
        thresholds[i] = line.unwrap().parse::<f64>().expect("Cannot parse String into f64!");
    }

    thresholds
}

/*
This function converts a line of the input file into a Vec of u8.
 */
fn convert_line_into_u8(line: String) -> Vec<u8> {
    line.chars()
        .map(|ch| ch.to_digit(10).unwrap() as u8)
        .collect::<Vec<u8>>()
}

fn get_current_dir() -> String {
    let current_dir = std::env::current_dir().expect("Failed to get current directory");

    // Convert the PathBuf to a string
    let current_dir_str = current_dir.to_string_lossy();

    // Replace backslashes with forward slashes
    let c = '\\';
    current_dir_str.replace(c, "/")
}

fn read_zip(path: &String) -> std::io::Result<()> {
    let extract_to_dir = format!("{path}/simulation/");
    let zip_file_path = format!("{extract_to_dir}inputSpikes.zip");

    let file = File::open(zip_file_path)?;
    let mut archive = ZipArchive::new(file)?;

    let mut file = archive.by_index(0)?;
    let output_path = format!("{extract_to_dir}inputSpikes.txt");

    let mut outfile = File::create(output_path)?;
    std::io::copy(&mut file, &mut outfile)?;

    Ok(())
}
