#include "extractiondata.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <list>
#include <iostream>
#include <vector>
#include <fstream>

/*Esta parte correponde a la primera funcion miembro
 *Se realiza la lectura de fichero csv
 *En este sector del programa se almacena en un vector de vectores del tipo string */

std::vector<std::vector<std::string>> ExtractionData::LeerCSV(){
    /*En principio se abre y se almacena el fichero en un buffer temporal "archivo"*/
    std::fstream archivo(dataset);
    /*A continuación se creará un vector de vectores del tipo string*/
    std::vector<std::vector<std::string>> datosString;
    /*el objetivo es recorrer cada linea del fichero y enviarla como vector al vector de vectores del tipo string*/
    std::string linea = "";
    while(getline(archivo,linea)){
        std::vector<std::string> vector;
        /*Como prosiguiente se identifica cada elemento que compone el vector, lo que divide o segmenta
         * cada elemeno con boost*/
        boost::algorithm::split(vector, linea, boost::is_any_of(delimitador));
        /*Y Finalmente se ingresa al buffer temporal*/
        datosString.push_back(vector);
    }
    /*En esta parte se cierra el fichero csv*/
    archivo.close();
    /*A continuación se retorna el vector de vectores*/
    return datosString;
}

/*Segunda funcion miembro
 * Pasar el vector de vectores del tipo string a un objeto del tipo Eigen, para las correspondientes operaciones*/

Eigen::MatrixXd ExtractionData::CSVtoEigen(
        std::vector<std::vector<std::string>> dataSet,
        int filas,
        int columnas){
    /*Se revisa si tiene o no cabecera*/
    if(header==true){
        filas = filas-1;
    }
    Eigen::MatrixXd matriz(columnas, filas);
    /*Se llena la matriz con los datos del dataset*/
    for(int i=1; i<filas; i++){
        for(int j=0; j<columnas; j++){
            /*Se pasa flotante el tipo String*/
            matriz(j,i) = atof(dataSet[i][j].c_str());
        }
    }
    /*Se retorna la matriz transpuesta*/
    return matriz.transpose();
}

/*Funcion para extraer el promedio */
/*Esta Función se usa para extraer el promedio, y cuando el programador no esta seguro cual es el
 * tipo de dato va a regresar la función, se utiliza auto "nombre_funcion" decltype
 */
auto ExtractionData::Promedio(Eigen::MatrixXd datos) -> decltype(datos.colwise().mean()){
    return datos.colwise().mean();
}

/*Funcion para extraer la desviacion estandar*/
auto ExtractionData::DevStand(Eigen::MatrixXd datos) -> decltype(((datos.array().square().colwise().sum()) / (datos.rows()-1)).sqrt()){
    //colwise - elemento de columnas
    return ((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt();
}

/*Funcion para normalizar los datos*/
/*
 * Se retorna la matriz de datos normalizada y la funcion recibe como argumentos la matriz de datos
 */
Eigen::MatrixXd ExtractionData::Norm(Eigen::MatrixXd datos){
    /*Se escalan los datos xi-mean*/
    Eigen::MatrixXd math_escalar = datos.rowwise()-Promedio(datos);
    /*Se calcula la normalizacion*/
    Eigen::MatrixXd math_normal = math_escalar.array().rowwise()/DevStand(math_escalar);

    return math_normal;
}

/*Funcion para dividir en 4 grandes grupos
 * X_train, y_train, X_test, y_test
*/
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> ExtractionData::TrainTestSplit(Eigen::MatrixXd datos, float size_train){
    /*Cantidad de filas totales*/
    int filas_totales = datos.rows();
    /*Cantidad de filas para entrenamiento*/
    int filas_train = round(filas_totales*size_train);
    /*Cantidad de filas para prueba*/
    int filas_test = filas_totales-filas_train;
    Eigen::MatrixXd Train = datos.topRows(filas_train);
    /*Se desprenden para independientes y dependientes*/
    Eigen::MatrixXd X_train = Train.leftCols(datos.cols()-1);
    Eigen::MatrixXd y_train = Train.rightCols(1);

    Eigen::MatrixXd Test = datos.bottomRows(filas_test);

    Eigen::MatrixXd X_test = Test.leftCols(datos.cols()-1);
    Eigen::MatrixXd y_test = Test.rightCols(1);

    /*Se completa la tupla y se retorna*/
    return std::make_tuple(X_train, y_train, X_test, y_test);
}


void ExtractionData::VectortoFile(std::vector<float> vector, std::string file_name){
    std::ofstream file_salida(file_name);
    //Se crea un iterador para almacenar la salida del vector
    std::ostream_iterator<float> salida_iterador(file_salida,"\n");
    //Se copia cada valor desde el inicio hasta el fin del iterator en el fichero
    std::copy(vector.begin(),vector.end(),salida_iterador);
}

// Para efectos de manipulacion y visualzacion se crea la funcion matriz eigen a fichero
void ExtractionData::EigentoFile(Eigen::MatrixXd matriz, std::string file_name){
    std::ofstream file_salida(file_name);
    if(file_salida.is_open()){
        file_salida << matriz << "\n";
    }
}
