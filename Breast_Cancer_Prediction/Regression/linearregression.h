#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

class linearregression
{
public:
    float F_OLS_Costo(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::MatrixXd thetas);
    std::tuple<Eigen::VectorXd, std::vector<float>> GradientDescent(Eigen::MatrixXd X,
                                                                                      Eigen::MatrixXd Y,
                                                                                      Eigen::MatrixXd thetas,
                                                                                      float alpha,
                                                                                      int num_iter);
    float R2_Score(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);
};

#endif // LINEARREGRESSION_H
