#pragma once

#include <Graph.hpp>

#define BOOLEAN_NN_EPS 1e-6

class BooleanNeuralNetwork {
	private:
    using ConnectionGraph = Graph<double, int>;
		uint32_t system_size;

	public:
    ConnectionGraph connections;

    BooleanNeuralNetwork()=default;
    BooleanNeuralNetwork(size_t system_size) : system_size(system_size) {
      connections = ConnectionGraph(system_size);
    }

    void randomize(std::minstd_rand& rng) {
      for (size_t i = 0; i < system_size; i++) {
        double d = (double) rng() / RAND_MAX;
        connections.set_val(i, (d < 0.5) ? 1 : -1);
      }
    }

    std::vector<int> onsite_potential() const {
      std::vector<double> vals(system_size, 0.0);
      for (size_t i = 0; i < system_size; i++) {
        for (auto const& [j, c] : connections.edges[i]) {
          vals[i] += c*connections.get_val(j);
        }
      }

      std::vector<int> _vals(system_size);
      for (size_t i = 0; i < system_size; i++) {
        if (std::abs(vals[i]) < BOOLEAN_NN_EPS) {
          _vals[i] = 0;
        } else {
          _vals[i] = std::signbit(vals[i]) ? -1 : 1;
        }
      }

      return _vals;
    }

    void update() {
      auto f = onsite_potential();
      for (size_t i = 0; i < system_size; i++) {
        if (f[i] != 0) {
          connections.set_val(i, f[i]);
        }
      }
    }

    void update(double eta, std::minstd_rand& rng) {
      auto f = onsite_potential();
      for (size_t i = 0; i < system_size; i++) {
        if (f[i] != 0) {
          double d = (double) rng() / RAND_MAX;
          int sign = (d < eta) ? -1 : 1;
          connections.set_val(i, sign*f[i]);
        }
      }
    }

    double get_order() const {
      double s = 0.0;
      for (size_t i = 0; i < system_size; i++) {
        s += connections.get_val(i);
      }

      return s/system_size;
    }
};
