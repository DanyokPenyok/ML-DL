#include "deepmnist.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <random>
using namespace std;
deepmnist::deepmnist(const vector<vector<double>>& initX, int hidenneuronscount, int outneutonscount, double lr, int epoches) :X(initX), hidenneuronsC(hidenneuronscount), outneuronsC(outneutonscount), learningrate(lr), epoch(epoches)
{
	vectorXsize = X.size();
	vvectorXsize = X[0].size() - 1;
	whide.resize(hidenneuronscount, vector<double>(vvectorXsize));
	wout.resize(outneutonscount, vector<double>(hidenneuronscount));
	biasH.resize(hidenneuronscount);
	biasO.resize(outneutonscount);
	Xtrue.resize(vectorXsize);
	H.resize(hidenneuronsC);
	out.resize(outneuronsC);
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> dist(-0.05, 0.05);
	for (size_t i = 0; i < vectorXsize; i++)
	{
		Xtrue[i] = X[i][vvectorXsize];
	}
	for (size_t i = 0; i < hidenneuronscount; i++)
	{
		for (size_t j = 0; j < vvectorXsize; j++)
		{
			whide[i][j] = dist(gen);
		}
		biasH[i] = dist(gen);
	}
	for (size_t i = 0; i < outneutonscount; i++)
	{
		for (size_t j = 0; j < hidenneuronscount; j++)
		{
			wout[i][j] = dist(gen);
		}
		biasO[i] = dist(gen);
	}
}  

double deepmnist::forward(int row) {
	//hide
	for (size_t i = 0; i < hidenneuronsC; i++)
	{
		double sum = 0;
		for (size_t j = 0; j < vvectorXsize; j++)
		{
			sum += X[row][j] * whide[i][j];
		}
		sum += biasH[i];
		H[i] = 1.0 / (1.0 + exp(-sum));
	}
	//out
	for (size_t i = 0; i < outneuronsC; i++)
	{
		double sum = 0;
		for (size_t j = 0; j < hidenneuronsC; j++)
		{
			sum += H[j] * wout[i][j];
		}
		sum += biasO[i];
		out[i] = 1.0 / (1.0 + exp(-sum));
	}
	return 0;
}

void deepmnist::backward(int row) {
	vector<double> deltaout(outneuronsC);
	for (size_t i = 0; i < outneuronsC; i++)
	{
		double Xtrue10 = (i == Xtrue[row]) ? 1.0 : 0.0;
		double error = Xtrue10 - out[i];
		deltaout[i] = error * out[i] * (1 - out[i]);
	}
	vector<double> deltahiden(hidenneuronsC);
	for (size_t i = 0; i < hidenneuronsC; i++)
	{
		double sum = 0;
		for (size_t j = 0; j < outneuronsC; j++)
		{
			sum += deltaout[j] * wout[j][i];
		}
		deltahiden[i] = sum * H[i] * (1 - H[i]);
		
	}
	//выход
	for (size_t i = 0; i < outneuronsC; i++)
	{
		for (size_t j = 0; j < hidenneuronsC; j++)
		{
			wout[i][j] = wout[i][j] + (learningrate * deltaout[i] * H[j]);
		}
		biasO[i] = biasO[i] + (learningrate * deltaout[i]);
	}
	//вход
	for (size_t i = 0; i < hidenneuronsC; i++)
	{
		for (size_t j = 0; j < vvectorXsize; j++)
		{
			whide[i][j] = whide[i][j] + (learningrate * deltahiden[i] * X[row][j]);
		}
		biasH[i] = biasH[i] + (learningrate * deltahiden[i]);
	}
}

void deepmnist::learning() {
	for (size_t i = 0; i < epoch; i++)
	{
		for (size_t j = 0; j < vectorXsize; j++)
		{
			forward(j);
			backward(j);
		}
	}
}
void deepmnist::printall() {
	for (size_t i = 0; i < vectorXsize; i++)
	{
		cout << "elementarray " << i + 1 << " xtrue: " << Xtrue[i] << " return(out[i]): ";
		forward(i);
		for (size_t j = 0; j < out.size(); j++)
		{
			cout << out[j] << endl;
		}
	}
}

void deepmnist::print(int row) {
	for (size_t i = 0; i < vvectorXsize; i++)
	{
		if (X[row][i] > 0.6) {
			cout << "11";
		}
		else if (X[row][i] > 0.1) {
			cout << "::";
		}
		else {
			cout << "  ";
		}
		if ((i + 1) % 28 == 0)
		{
			cout << endl;
		}
	}
	cout << endl;

	forward(row);
	cout << "Xtrue " << Xtrue[row] << endl;
	for (size_t i = 0; i < outneuronsC; i++)
	{
		cout << "chance " << i << " " << out[i] <<endl;
	}
}