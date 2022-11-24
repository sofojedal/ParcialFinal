#include "linearregression.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

/*
 * En primer lugar se explicará la función de costo para la regresión lineal, la cual es
 * basada en los minimos cuadrados ordinarios.
 *
 * Para luego entrenar el modelo, lo cual implica
 * minimizar la función de costo, y de esta forma poder medir la precisión de la función de hipotesis.
 *
 * En segundo lugar, una función de costo es la forma de penalizar el modelo por cometer un error,
 * de esta forma se implementauna función de tipo flotante, la cual toma como entrada los valores
 * tanto de x, como de y.
 */

float linearregression::F_OLS_Costo(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::MatrixXd thetas){
    //Eigen::MatrixXd diferencia = pow((X*thetas - Y).array(),2);
    //return (diferencia.sum() / (2*X.rows()));
    Eigen::MatrixXd diferencia = pow((X*thetas - Y).array(),2);
    return (diferencia.sum() / (2*X.rows()));
}

/*
 * En segundo lugar se explicará la función de optimización, la cual se comprende de la función de
 * gradiente descendiente, la cual avanza hasta encontrar el punto minimo que representa el valor e
 * l valor optimo para la función que necesita proveer al programa para que este pueda dar al
 * algoritmo un valor inicial para theta, la cual representa el cambio iterativamente hasta que este converja
 * el valor al minimo de la función de costo. Donde basicamente esta describe el gradiente descendiente,
 * la idea es calcular el gradiente para la función de costo, el cual es dado por la derivada parcial de
 * la función; esta tendrá un alpha que representa el salto del gradiente, las entradas para la función seran X, Y,
 * theta, alpha y la cantidad de veces que son necesarias para actualizar theta hasta la función converja.
 */

std::tuple<Eigen::VectorXd, std::vector<float>> linearregression::GradientDescent(Eigen::MatrixXd X,
                                                                                  Eigen::MatrixXd Y,
                                                                                  Eigen::MatrixXd thetas,
                                                                                  float alpha,
                                                                                  int num_iter){
    /*En esta parte se indica el almacenamiento temporal de parametros theta*/
    Eigen::MatrixXd temporal = thetas;
    /*Esta variable indica la cantidad de parametros (m) features*/
    int parametros = thetas.rows();
    /*Ubicacion costo inicial, que se actualiza con los nuevos pesos*/
    std::vector<float> costo;
    //En costo ingresaremos los valores de la funcion de costo
    costo.push_back(F_OLS_Costo(X,Y,thetas));

    /*
     * En esta parte se iterará segun el numero de iteraciones y el ratio de aprendizaje para encontrar los valores optimos
     * Por cada iteracion se calcula la funcion de error que se usa para multiplicar cada deature para obtener
     * el error de cada feature y asi almacenarlo en la variable tem. Se actualiza theta y se calcula el nuevo
     * valor de la funcion de costo basada en el nuevo valor de theta
     */
    for(int i=0; i<num_iter; i++){
        Eigen::MatrixXd error = X*thetas-Y;
        for(int j=0; j<parametros; j++){
            Eigen::MatrixXd X_i = X.col(j);
            Eigen::MatrixXd termino = error.cwiseProduct(X_i);
            temporal(j,0) = thetas(j,0) - ((alpha/X.rows())*termino.sum()); //((alpha/X.rows())*termino.sum());
        }
        thetas = temporal;
        //Esta linea indica que ingresaremos los valores de la funcion de costo
        costo.push_back(F_OLS_Costo(X,Y,thetas));
    }
    return std::make_tuple(thetas,costo);
}

/*
 * En tercer lugar, la metrica de rendimiento, presenta que tan bueno es nuestro proyecto,
 * se procede a crear la metrica de redimiento.
 * En el caso de R² score, el coeficiente de determinación, es donde el mejor valorposible es 1.
 * Por otro lado la métrica de evaluación se tiene el R2, la cual representa una medida de que
 * tan bueno es nuestro modelo.
*/

float linearregression::R2_Score(Eigen::MatrixXd y, Eigen::MatrixXd y_hat){
     auto numerador = pow((y-y_hat).array(),2).sum();
     auto denominador = pow(y.array() - y.mean(),2).sum();
     return (1 - (numerador/denominador));
}
