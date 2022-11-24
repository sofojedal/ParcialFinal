#ifndef EXTRACTIONDATA_H
#define EXTRACTIONDATA_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <list>
#include <iostream>
#include <vector>
#include <fstream>

/*
 * En este programa basicamente se realizará una extracción de datos, donde se leera un fichero csv,
 * se entrarán los argumentos a la clase, en este caso estariamos hablando del lugar del dataset (csv),
 * el separador, y se valida si este tiene encabezado o no, es decir el titulo de cada columna.
 *
 * Este también pasa a un vetor de vectores de tipo string, así mismo pasa a el vector de vectores string
 * a Eigen, este también comprende el promedio, desviación, normalización y métricas.*/


class ExtractionData
{
    /*Argumentos de entrada a la clase*/

    //Corresponde a la ruta del dataset
    std::string dataset;
    //Corresponde al separador entre datos
    std::string delimitador;
    //Corresponde al uso  de cabecera o no
    bool header;

public:

    /* En esta parte se crea el constructor con los argumentos de entrada, tenemos el vector de vectores de tipo string */
    ExtractionData(std::string data,
                   std::string separador,
                   bool cabecera):
        dataset(data), delimitador(separador), header(cabecera){}

    /*Corresponde al prototipo de metodos y funciones*/
    std::vector<std::vector<std::string>> LeerCSV();
    Eigen::MatrixXd CSVtoEigen(
            std::vector<std::vector<std::string>> dataSet,
            int filas,
            int columnas);
    auto Promedio(Eigen::MatrixXd datos) -> decltype(datos.colwise().mean());
    auto DevStand(Eigen::MatrixXd datos) -> decltype(((datos.array().square().colwise().sum()) / (datos.rows()-1)).sqrt());
    Eigen::MatrixXd Norm(Eigen::MatrixXd datos);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> TrainTestSplit(Eigen::MatrixXd datos, float size_train);

    void VectortoFile(std::vector<float> vector, std::string file_name);
    void EigentoFile(Eigen::MatrixXd matriz, std::string file_name);
};

#endif // EXTRACTIONDATA_H
