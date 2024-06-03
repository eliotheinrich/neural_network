#pragma once

#include <Frame.h>

#include "BooleanNeuralNetwork.hpp"

class NonlocalNeuralSimulator: public dataframe::Simulator {
	private:
    BooleanNeuralNetwork state;

		uint32_t system_size;
    double eta;
    double alpha;

    bool sample_configurations;


    double rand_pl(int i, int j) {
      double dist1 = std::abs(i - j);
      double dist2 = system_size - dist1;
      double dist = std::min(dist1, dist2)/system_size;

      double N = (std::pow(system_size/2.0, alpha + 1.0) - 1.0)/(1.0 + alpha);
      double p = std::pow(dist, alpha)/N;
      if (p > 1.0 || p < 0.0) {
        throw std::runtime_error(
          fmt::format(
            "Error calculating link probability: i = {}, j = {}, p = {}",
            i, j, p
          )
        );
      }

      return p;
    }

    void reset_connections() {
      for (int i = 0; i < system_size; i++) {
        for (int j = 0; j < system_size; j++) {
          if (i == j) {
            continue;
          }

          double p = rand_pl(i, j);
          bool edge_should_exist = randf() < p;
          if (state.connections.contains_edge(i, j)) {
            if (!edge_should_exist) { // Edge exists but it should not
              state.connections.remove_edge(i, j);
            }
            // Otherwise, edge exists and it should exist
          } else {
            if (edge_should_exist) { // Edge does not exist but it should
              state.connections.add_directed_edge(i, j, 1.0);
            }
            // Otherwise, edge does not exist and it should not
          }
        }
      }
    }

    void init_state() {
      state = BooleanNeuralNetwork(system_size);
      reset_connections();
      state.randomize(rng);
    }

	public:
    NonlocalNeuralSimulator(dataframe::Params &params, uint32_t num_threads) : dataframe::Simulator(params) {
      system_size = dataframe::utils::get<int>(params, "system_size");
      eta = dataframe::utils::get<double>(params, "eta");
      alpha = dataframe::utils::get<double>(params, "alpha", 2.0);

      sample_configurations = dataframe::utils::get<int>(params, "sample_configurations", false);

      init_state();
    }

		virtual void equilibration_timesteps(uint32_t num_steps) override {
			timesteps(num_steps);
		}

    virtual void timesteps(uint32_t num_steps) override {
      for (uint32_t t = 0; t < num_steps; t++) {
        state.update(eta, rng);
        reset_connections();
      }
    }

    void add_order_sample(dataframe::data_t& samples) {
      double s = state.get_order();
      dataframe::utils::emplace(samples, "order", std::abs(s));
    }

    void add_configuration_samples(dataframe::data_t& samples) {
      std::vector<double> _state(state.connections.num_vertices);
      for (size_t i = 0; i < state.connections.num_vertices; i++) {
        _state[i] = state.connections.vals[i];
      }

      dataframe::utils::emplace(samples, "spins", _state);
    }

		virtual dataframe::data_t take_samples() override {
      dataframe::data_t samples;

      add_order_sample(samples);

      if (sample_configurations) {
        add_configuration_samples(samples);
      }

      return samples;
    }
};
