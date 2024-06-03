#pragma once

#include <Frame.h>
#include <stdexcept>

#include "BooleanNeuralNetwork.hpp"

#define NNS_UNIFORM 0

class LatticeNeuralSimulator: public dataframe::Simulator {
	private:
    BooleanNeuralNetwork state;

		uint32_t system_size;
    double eta;
    double p;
    int connection_distribution;
    int dim;
    bool obc;

    bool sample_configurations;

    double sample_connection_strength() {
      if (connection_distribution == NNS_UNIFORM) {
        return 1.0;
      } else {
        return 0.0;
      }
    }

    size_t mod(int i) const {
      int L = static_cast<int>(system_size);
      return (i % L + L) % L;
    }

    std::pair<int, int> to_coordinates(size_t i) const {
      int x = i % system_size;
      int y = i / system_size;
      return std::make_pair(x, y); 
    }

    size_t to_index(int x, int y) const {
      return mod(x) + mod(y) * system_size;
    }

    void init_2d_state() {
      size_t N = system_size*system_size;

      state = BooleanNeuralNetwork(N);

      for (size_t i = 0; i < N; i++) {
        auto [x, y] = to_coordinates(i);
        if (obc && (y == 0 || y == system_size - 1)) {
          continue;
        }
        size_t i1 = to_index(x+1, y);
        size_t i2 = to_index(x-1, y);
        size_t i3 = to_index(x, y+1);
        size_t i4 = to_index(x, y-1);

        std::vector<size_t> inds{i1, i2, i3, i4};

        if (randf() < p) {
          for (auto j : inds) {
            state.connections.add_directed_edge(j, i, sample_connection_strength());
          }
        }
      }
    }

    void init_1d_state() {
      size_t N = system_size;

      state = BooleanNeuralNetwork(N);

      for (int x = 0; x < N; x++) {
        if (obc && (x == 0 || x == system_size - 1)) {
          continue;
        }

        size_t i1 = mod(x + 1);
        size_t i2 = mod(x - 1);

        std::vector<size_t> inds{i1, i2};

        if (randf() < p) {
          for (auto j : inds) {
            state.connections.add_directed_edge(j, x, sample_connection_strength());
          }
        }
      }
    }

    void init_state() {
      if (dim == 1) {
        init_1d_state();
      } else if (dim == 2) {
        init_2d_state();
      } else {
          throw std::runtime_error("State dim must be 1 or 2.");
      }

      state.randomize(rng);
    }

	public:
    LatticeNeuralSimulator(dataframe::Params &params, uint32_t num_threads) : dataframe::Simulator(params) {
      system_size = dataframe::utils::get<int>(params, "system_size");
      eta = dataframe::utils::get<double>(params, "eta");
      p = dataframe::utils::get<double>(params, "p");
      connection_distribution = dataframe::utils::get<int>(params, "connection_distribution", NNS_UNIFORM);
      obc = dataframe::utils::get<int>(params, "obc", false);

      dim = dataframe::utils::get<int>(params, "dim", 2);

      sample_configurations = dataframe::utils::get<int>(params, "sample_configurations", false);

      init_state();
    }

		virtual void equilibration_timesteps(uint32_t num_steps) override {
			timesteps(num_steps);
		}

    virtual void timesteps(uint32_t num_steps) override {
      for (uint32_t t = 0; t < num_steps; t++) {
        state.update(eta, rng);
      }
    }

    void add_order_sample(dataframe::data_t& samples) {
      double s = state.get_order();
      dataframe::utils::emplace(samples, "order", s);
      dataframe::utils::emplace(samples, "order_abs", std::abs(s));
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
