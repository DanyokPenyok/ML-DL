#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <windows.h>
#include "ainums.h"
#include "ainumsdeep.h"
#include "deepmnist.h"  
using namespace std;

vector<vector<double>> parse_mnist_csv(const string & filename) {
    vector<vector<double>> dataset;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "не удалось открыть файл " << filename << endl;
        return dataset;
    }
    string line;
    int row_count = 0;
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> row;
        while (getline(file, line)) {
            stringstream ss(line);
            string value;
            vector<double> row;
            while (getline(ss, value, ',')) {
                if (!value.empty()) {
                    row.push_back(stod(value));
                }
            }
            if (row.empty()) continue;
            if (row.size() < 10) {
                continue;
            }
            double target_label = row.back();
            row.pop_back();
            for (size_t i = 0; i < row.size(); i++) {
                row[i] /= 255.0;
            }
            row.push_back(target_label);
            dataset.push_back(row);
            row_count++;
            if (row_count % 500 == 0) {
                cout << "загружено " << row_count << endl;
            }
        }
    }

    file.close();
    cout << "загружено картинок: " << dataset.size() << endl;
    return dataset;
}
int main()
{
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);
    cout << "пример работы мо" << endl;
    vector<vector<double>> init = {
        {0, 0, 0, 0,    0},
        {0, 0, 0, 1,    0},
        {0, 1, 0, 0,    0},
        {1, 0, 0, 0,    0},
        {0, 0, 1, 1,    1},
        {0, 1, 0, 1,    1},
        {0, 1, 1, 0,    1},
        {1, 0, 0, 1,    1},
        {1, 0, 1, 0,    1},
        {1, 1, 0, 0,    1},
        {0, 1, 1, 1,    1},
        {1, 0, 1, 1,    1},
        {1, 1, 0, 1,    1},
        {1, 1, 1, 0,    1},
        {1, 1, 1, 1,    1}
    };
    ainums learn(init, 0.5, 100000);
    learn.learning();
    learn.print();
    cout << "\nпример работы мо на малом наборе\n";
    vector<vector<double>> xinit = {
    {0, 0,   0}, 
    {1, 0,   1}, 
    {0, 1,   1},
    {1, 1,   0}
    };
    ainums my_network(xinit, 0.5, 20000);
    my_network.learning();
    my_network.print();
    cout << "\nпример работы го на малом наборе\n";
    ainumsdeep my_deepnetwork(xinit, 100000, 0.5, 4,1);
    my_deepnetwork.learning();
    my_deepnetwork.print();
    cout << "\nпример работы го на mnist\n";
    string filename = "C:\\Users\\danya\\OneDrive\\Рабочий стол\\mnist_1000.csv";
    vector<vector<double>> dataset = parse_mnist_csv(filename);
    deepmnist my_networkmnist(dataset, 32, 10, 0.05, 1000);
    my_networkmnist.learning();
    my_networkmnist.print(0);
}
