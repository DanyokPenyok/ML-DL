#pragma once
#include <cmath>
#include <vector>
#include <iostream>
#include <random>
using namespace std;
class deepmnist
{
private:
	const vector<vector<double>>& X;
	vector <double> Xtrue;
	size_t vectorXsize;
	size_t vvectorXsize;
	int hidenneuronsC;
	int outneuronsC;
	double learningrate;
	int epoch;
	vector<vector<double>> whide;
	vector<vector<double>> wout;
	vector<double> biasH;
	vector<double> biasO;
	vector<double> H;
	vector<double>out;
public:
	deepmnist(const vector<vector<double>> & initX, int hidenneuronscount,int outneutonscount, double lr, int epoches);
	double forward(int row);
	void backward(int row);
	void learning();
	void print(int row);
	void printall();
};

