#ifndef H_DATA
#define H_DATA


#include <signal.h>
#include <random>
#include <chrono>
#include <fstream>
#include <iostream>
#include <functional>
#include <sys/stat.h>
#ifdef NPY
#include "cnpy.h"
#endif
#include "plotter.h"


void indices2buffer(const std::string& dataset_dir, std::vector<std::string>& data_indices, bool& break_sim);
void feed_network(const std::string& dataset_dir, std::vector<std::string>& dataset, float sim_int, float sim_step,
                  const int sim_step_range[2], int sim_num_steps, float scale_ets, Network *&SNN,
                  plotterGL *plotter, bool openGL, bool data_augmentation, bool record_activity, 
                  std::string snapshots_dir, std::string weights_out, bool random, bool& break_sim);
void weights_to_csv(std::string& foldername, Network *&SNN, std::string dir = "");
void csv_to_weights(std::string& folder, Network *&SNN, plotterGL *&plotter);
void activity_to_npy(Network *&SNN, std::string snapshot_dir, std::string folder, int run, int step,
                     plotterGL *plotter);

#endif