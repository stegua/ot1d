/*
 * @fileoverview Copyright (c) 2021, Stefano Gualandi,
 *               via Ferrata, 5, I-27100, Pavia, Italy
 *
 * @author stefano.gualandi@gmail.com (Stefano Gualandi)
 *
 *
 * External libraries:
 * - Super fast sorting algorithm (C++11) 
 *    https://github.com/orlp/pdqsort
 *    (Zlib license)
 *
 * - Efficient parallel sorting algorithm (C++11)
 *    https://github.com/baserinia/parallel-sort
 *    (GPL-3.0 License)
 *    
 *
 * NOTE: 
 *   - https://encyclopediaofmath.org/index.php?title=Wasserstein_metric
 *   - https://www.euroscipy.org/2018/descriptions/Detecting%20anomalies%20using%20statistical%20distances.html
 *   - https://www.kaggle.com/nhan1212/some-statistical-distances/comments
 *   - https://www.datadoghq.com/blog/engineering/robust-statistical-distances-for-machine-learning/
 */


#include "OT1D.hpp"

#include <random>

int main(int argc, char* argv[]) {

    int seed = 13;

    std::random_device
      rd; // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(seed); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> Uniform01(0, 1);
    std::uniform_real_distribution<> Uniform12(1, 2);

    int N = 1000000;

    std::vector<double> x;
    std::vector<double> y;
    x.reserve(N);
    y.reserve(N);
    
    for (int i = 0; i < N; i++) {
        x.push_back(Uniform01(gen));
        y.push_back(Uniform12(gen));
    }

    double z = OT1Da0(N, &x[0], &y[0], true, 16);

    fprintf(stdout, "Wasserstein distance of order 1: W1(x,y) = %.6f\n", z);

    return 0;
}